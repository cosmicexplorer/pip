from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pip._internal.models.candidate import InstallationCandidate
from pip._internal.utils._log import getLogger
from pip._internal.utils.hashes import Hashes

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self


logger = getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CandidateHashFilter:
    project_name: str
    hashes: Hashes | None

    @dataclass(frozen=True)
    class HashMatches:
        match_count: int
        matches_or_no_digest: tuple[InstallationCandidate, ...]
        non_matches: tuple[InstallationCandidate, ...]
        candidates: tuple[InstallationCandidate, ...]

        __slots__ = [
            "match_count",
            "matches_or_no_digest",
            "non_matches",
            "candidates",
            "__dict__",
        ]

        @functools.cached_property
        def filtered(self) -> tuple[InstallationCandidate, ...]:
            if self.match_count:
                return self.matches_or_no_digest
            return self.candidates

        @dataclass(frozen=True)
        class NonMatches:
            non_matches: tuple[InstallationCandidate, ...]

            def __str__(self) -> str:
                n = len(self.non_matches)
                if not n:
                    return "discarding no candidates"
                joined = "\n  ".join(str(c.link) for c in self.non_matches)
                return f"discarding {n} non-matches:\n  {joined}"

        def discarded_candidates(self) -> NonMatches:
            if len(self.filtered) == len(self.candidates):
                return self.__class__.NonMatches(())
            return self.__class__.NonMatches(self.non_matches)

        @classmethod
        def calculate(
            cls, hashes: Hashes, candidates: tuple[InstallationCandidate, ...]
        ) -> Self:
            matches_or_no_digest = []
            # Collect the non-matches for logging purposes.
            non_matches = []
            match_count = 0
            for candidate in candidates:
                link = candidate.link
                if not link.has_hash:
                    pass
                elif link.is_hash_allowed(hashes=hashes):
                    match_count += 1
                else:
                    non_matches.append(candidate)
                    continue
                matches_or_no_digest.append(candidate)
            return cls(
                match_count=match_count,
                matches_or_no_digest=tuple(matches_or_no_digest),
                non_matches=tuple(non_matches),
                candidates=candidates,
            )

    def filter_unallowed_hashes(
        self,
        candidates: Iterable[InstallationCandidate],
    ) -> tuple[InstallationCandidate, ...]:
        """
        Filter out candidates whose hashes aren't allowed, and return a new
        list of candidates.

        If at least one candidate has an allowed hash, then all candidates with
        either an allowed hash or no hash specified are returned.  Otherwise,
        the given candidates are returned.

        Including the candidates with no hash specified when there is a match
        allows a warning to be logged if there is a more preferred candidate
        with no hash specified.  Returning all candidates in the case of no
        matches lets pip report the hash of the candidate that would otherwise
        have been installed (e.g. permitting the user to more easily update
        their requirements file with the desired hash).
        """
        # Make sure we're not returning back the given value.
        candidates = tuple(candidates)
        if not self.hashes:
            logger.debug(
                "Given no hashes to check %s links for project %r: "
                "discarding no candidates",
                len(candidates),
                self.project_name,
            )
            return candidates
        assert self.hashes is not None

        hash_matches = self.__class__.HashMatches.calculate(self.hashes, candidates)

        logger.debug(
            "Checked %s links for project %r against %s hashes "
            "(%s matches, %s no digest): %s",
            len(hash_matches.candidates),
            self.project_name,
            self.hashes.digest_count,
            hash_matches.match_count,
            len(hash_matches.matches_or_no_digest) - hash_matches.match_count,
            hash_matches.discarded_candidates(),
        )

        return hash_matches.filtered
