from typing import Optional


class ResponseLengthScorer:
    def __init__(
        self,
        baseline: float,
        cap: Optional[float] = None,
    ):
        """
        Score problems based on average student response length.
        
        Args:
            baseline: Reference length for neutral reward (1.0x). Longer responses get higher rewards.
            cap: Optional maximum length to prevent outliers from dominating.
        """
        self.baseline = baseline
        self.cap = cap
        
    def score(self, avg_length: float) -> float:
        """
        Calculate reward score based on average response length.
        
        Args:
            avg_length: Average length of student responses for a problem.
            
        Returns:
            Reward score (avg_length / baseline). Scaling handled by teacher_reward_weights.
        """
        if avg_length <= 0:
            return 0.0
            
        # Cap outliers if specified
        if self.cap:
            avg_length = min(avg_length, self.cap)
            
        # Normalize by baseline (scaling handled by teacher_reward_weights)
        return avg_length / self.baseline


class DifficultyScorer:
    def __init__(
        self,
        lower_limit,
        upper_limit,
        n_generations: int,
    ):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        scores = [i / n_generations for i in range(1, n_generations + 1)]
        self.valid_scores = [
            s for s in scores if lower_limit <= s <= upper_limit
        ]

    def score(self, solve_rate: float):
        """
        Calculate difficulty score using bi-directional normalized inverse gating.

        Bi-directional linear interpolation:

        lower_limit ---------------- midpoint ---------------- upper_limit
          y2 (=1/len)  ..........  y1 (=1.0)  ..........  y2 (=1/len)

        Any solve_rate outside the limits gets 0.
        """
        # Outside the gated interval → 0
        if solve_rate < self.lower_limit or solve_rate > self.upper_limit:
            return 0

        # Constants
        midpoint = (self.lower_limit + self.upper_limit) / 2
        half_range = midpoint - self.lower_limit  # == upper_limit - midpoint
        y1 = 1.0
        y2 = 1.0 / len(self.valid_scores)

        # Distance from the peak, normalised to [0, 1]
        distance = abs(solve_rate - midpoint)
        normalised_distance = distance / half_range if half_range else 0

        # Linear decay from y1 at the peak to y2 at the edges
        score = y1 - (y1 - y2) * normalised_distance
        return score
