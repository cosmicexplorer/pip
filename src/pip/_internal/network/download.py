"""Download files with progress indicators.
"""
import cgi
import io
import logging
import mimetypes
import os
from multiprocessing import Process
from multiprocessing import Pool as ProcessPool
from multiprocessing import Queue as ProcessQueue
from queue import Queue as ThreadQueue
from threading import Thread
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

from pip._vendor.requests.adapters import HTTPAdapter
from pip._vendor.requests.models import CONTENT_CHUNK_SIZE, Response

from pip._internal.cli.progress_bars import get_download_progress_renderer
from pip._internal.exceptions import NetworkConnectionError
from pip._internal.models.index import PyPI
from pip._internal.models.link import Link
from pip._internal.network.cache import is_from_cache
from pip._internal.network.session import PipSession
from pip._internal.network.utils import HEADERS, raise_for_status, response_chunks
from pip._internal.utils.misc import format_size, redact_auth_from_url, splitext

logger = logging.getLogger(__name__)


def _get_http_response_size(resp: Response) -> Optional[int]:
    try:
        return int(resp.headers["content-length"])
    except (ValueError, KeyError, TypeError):
        return None


def _prepare_download(
    resp: Response,
    link: Link,
    progress_bar: str,
) -> Iterable[bytes]:
    total_length = _get_http_response_size(resp)

    if link.netloc == PyPI.file_storage_domain:
        url = link.show_url
    else:
        url = link.url_without_fragment

    logged_url = redact_auth_from_url(url)

    if total_length:
        logged_url = "{} ({})".format(logged_url, format_size(total_length))

    if is_from_cache(resp):
        logger.info("Using cached %s", logged_url)
    else:
        logger.info("Downloading %s", logged_url)

    if logger.getEffectiveLevel() > logging.INFO:
        show_progress = False
    elif is_from_cache(resp):
        show_progress = False
    elif not total_length:
        show_progress = True
    elif total_length > (40 * 1000):
        show_progress = True
    else:
        show_progress = False

    chunks = response_chunks(resp, CONTENT_CHUNK_SIZE)

    if not show_progress:
        return chunks

    renderer = get_download_progress_renderer(bar_type=progress_bar, size=total_length)
    return renderer(chunks)


def sanitize_content_filename(filename: str) -> str:
    """
    Sanitize the "filename" value from a Content-Disposition header.
    """
    return os.path.basename(filename)


def parse_content_disposition(content_disposition: str, default_filename: str) -> str:
    """
    Parse the "filename" value from a Content-Disposition header, and
    return the default filename if the result is empty.
    """
    _type, params = cgi.parse_header(content_disposition)
    filename = params.get("filename")
    if filename:
        # We need to sanitize the filename to prevent directory traversal
        # in case the filename contains ".." path parts.
        filename = sanitize_content_filename(filename)
    return filename or default_filename


def _get_http_response_filename(resp: Response, link: Link) -> str:
    """Get an ideal filename from the given HTTP response, falling back to
    the link filename if not provided.
    """
    filename = link.filename  # fallback
    # Have a look at the Content-Disposition header for a better guess
    content_disposition = resp.headers.get("content-disposition")
    if content_disposition:
        filename = parse_content_disposition(content_disposition, filename)
    ext: Optional[str] = splitext(filename)[1]
    if not ext:
        ext = mimetypes.guess_extension(resp.headers.get("content-type", ""))
        if ext:
            filename += ext
    if not ext and link.url != resp.url:
        ext = os.path.splitext(resp.url)[1]
        if ext:
            filename += ext
    return filename


def _http_get_download(session: PipSession, link: Link) -> Response:
    target_url = link.url.split("#", 1)[0]
    resp = session.get(target_url, headers=HEADERS, stream=True)
    raise_for_status(resp)
    return resp


class Downloader:
    def __init__(
        self,
        session: PipSession,
        progress_bar: str,
    ) -> None:
        self._session = session
        self._progress_bar = progress_bar

    @property
    def session(self) -> PipSession:
        return self._session

    def get_http_response(self, link: Link) -> Response:
        try:
            return _http_get_download(self._session, link)
        except NetworkConnectionError as e:
            assert e.response is not None
            logger.critical(
                "HTTP error %s while getting %s", e.response.status_code, link
            )
            raise

    def download_chunks(self, resp: Response, link: Link) -> Iterator[bytes]:
        return _prepare_download(resp, link, self._progress_bar)

    def __call__(self, link: Link, location: str) -> Tuple[str, str]:
        """Download the file given by link into location."""
        resp = self.get_http_response(link)

        filename = _get_http_response_filename(resp, link)
        filepath = os.path.join(location, filename)

        chunks = self.download_chunks(resp, link)
        with open(filepath, "wb") as content_file:
            for chunk in chunks:
                content_file.write(chunk)
        content_type = resp.headers.get("Content-Type", "")
        return filepath, content_type


class BatchDownloader:
    def __init__(
        self,
        session: PipSession,
        progress_bar: str,
    ) -> None:
        self._downloader = Downloader(session, progress_bar)

    def _mount_pooled_connection_adapter(self) -> None:
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
        self._downloader.session.mount('https://', adapter)

    # def _split_link_responses(
    #     self, links: Iterable[Link]
    # ) -> Iterable[Tuple[Link, Tuple[str, Response]]]:
    #     download_responses: Dict[Link, Tuple[str, Response]] = {}

    #     for link in links:
    #         resp = self._downloader.get_http_response(link)
    #         content_type = resp.headers.get("Content-Type", "")
    #         download_responses[link] = content_type, resp

    #     return download_responses

    @staticmethod
    def multiprocess_target(
        link: Link, location: str,
        q: Union[ThreadQueue, ProcessQueue], downloader: Downloader
    ) -> None:
        filepath, content_type = downloader(link, location)
        q.put((link, (filepath, content_type)))

    def invoke_multiprocess(
        self, links: Iterable[Link], location: str,
        concurrency: int = 20,
        pool_connections: bool = False
    ) -> Iterable[Tuple[Link, Tuple[str, str]]]:
        if pool_connections:
            self._mount_pooled_connection_adapter()

        # pool = ProcessPool(concurrency)
        q = ProcessQueue(concurrency)
        processes: List[Process] = []
        for link in links:
            p = Process(target=self.multiprocess_target, args=(link, location, q, self._downloader))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            yield q.get()

    def invoke_threaded(
        self, links: Iterable[Link], location: str,
        concurrency: int = 20, pool_connections: bool = True
    ) -> Iterable[Tuple[Link, Tuple[str, str]]]:
        if pool_connections:
            self._mount_pooled_connection_adapter()

        q = ThreadQueue(concurrency)
        threads: List[Thread] = []
        for link in links:
            t = Thread(target=self.multiprocess_target, args=(link, location, q, self._downloader))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
            yield q.get()

    def invoke_parallel(
        self, links: Iterable[Link], location: str,
        pool_connections: bool = True
    ) -> Iterable[Tuple[Link, Tuple[str, str]]]:
        if pool_connections:
            self._mount_pooled_connection_adapter()

        download_locations: Dict[Link, Tuple[str, str]] = {}
        download_files: Dict[Link, io.BufferedWriter] = {}
        download_chunks: Dict[Link, Iterator[bytes]] = {}

        for link in links:
            resp = self._downloader.get_http_response(link)

            filename = _get_http_response_filename(resp, link)
            filepath = os.path.join(location, filename)

            content_type = resp.headers.get("Content-Type", "")

            download_locations[link] = (filepath, content_type)
            download_files[link] = open(filepath, 'wb')
            download_chunks[link] = self._downloader.download_chunks(resp, link)

        remaining_links = list(links)[:]
        while remaining_links:
            new_remaining_links = []
            for link in remaining_links:
                filepath, _ = download_locations[link]
                output_file = download_files[link]
                bytes_iterator = download_chunks[link]
                try:
                    chunk = next(bytes_iterator)
                    output_file.write(chunk)
                    new_remaining_links.append(link)
                except StopIteration:
                    output_file.close()
                    del download_files[link]
                    del download_chunks[link]
            remaining_links = new_remaining_links

        for link in links:
            filepath, content_type = download_locations[link]
            yield link, (filepath, content_type)

    def invoke_direct(
        self, links: Iterable[Link], location: str
    ) -> Iterable[Tuple[Link, Tuple[str, str]]]:
        for link in links:
            filepath, content_type = self._downloader(link, location)
            yield link, (filepath, content_type)

    def __call__(
        self, links: Iterable[Link], location: str
    ) -> Iterable[Tuple[Link, Tuple[str, str]]]:
        """Download the files given by links into location."""
        # return self.invoke_multiprocess(links, location)
        return self.invoke_threaded(links, location)
        # return self.invoke_parallel(links, location)
        # return self.invoke_direct(links, location)
