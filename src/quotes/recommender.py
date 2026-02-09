"""
Quote Recommender - Generate recommendation based on leveled quotes.
"""

from typing import Dict, List
from .leveler import LeveledComparison


class QuoteRecommender:
    """Generate recommendation from leveled quote comparison."""

    # Scoring weights
    WEIGHTS = {
        "price": 0.40,
        "warranty": 0.15,
        "completion_time": 0.15,
        "payment_terms": 0.10,
        "scope_completeness": 0.20,
    }

    def __init__(self):
        pass

    def recommend(self, leveled: LeveledComparison) -> Dict:
        """Generate recommendation from leveled comparison."""
        if not leveled.quotes:
            return {
                "lowest_bidder": "N/A",
                "recommended_bidder": "N/A",
                "lowest_amount": 0,
                "recommended_amount": 0,
                "reasons": ["No quotes to compare"],
                "risks": [],
            }

        # Find lowest bidder
        quotes_by_price = sorted(leveled.quotes, key=lambda q: q.leveled_total)
        lowest = quotes_by_price[0]

        # Score all quotes
        scores = {}
        for quote in leveled.quotes:
            scores[quote.subcontractor_name] = self._score_quote(quote, leveled)

        # Find recommended (highest score)
        recommended_name = max(scores, key=scores.get)
        recommended = next(q for q in leveled.quotes if q.subcontractor_name == recommended_name)

        # Generate reasons
        reasons = self._generate_reasons(recommended, lowest, leveled, scores)

        # Identify risks
        risks = self._identify_risks(recommended, leveled)

        # Calculate savings potential
        highest = max(q.leveled_total for q in leveled.quotes)
        savings = highest - recommended.leveled_total

        return {
            "lowest_bidder": lowest.subcontractor_name,
            "recommended_bidder": recommended.subcontractor_name,
            "lowest_amount": round(lowest.leveled_total, 2),
            "recommended_amount": round(recommended.leveled_total, 2),
            "highest_amount": round(highest, 2),
            "savings_potential": round(savings, 2),
            "scores": {name: round(score, 2) for name, score in scores.items()},
            "reasons": reasons,
            "risks": risks,
            "price_spread_pct": round((highest - lowest.leveled_total) / lowest.leveled_total * 100, 1),
        }

    def _score_quote(self, quote, leveled: LeveledComparison) -> float:
        """Calculate overall score for a quote."""
        scores = {}

        # Price score (inverse - lower is better)
        all_prices = [q.leveled_total for q in leveled.quotes]
        min_price = min(all_prices)
        max_price = max(all_prices)
        if max_price > min_price:
            price_score = 1 - ((quote.leveled_total - min_price) / (max_price - min_price))
        else:
            price_score = 1.0
        scores["price"] = price_score

        # Warranty score (more is better)
        all_warranties = [q.warranty_months for q in leveled.quotes]
        min_war = min(all_warranties)
        max_war = max(all_warranties)
        if max_war > min_war:
            warranty_score = (quote.warranty_months - min_war) / (max_war - min_war)
        else:
            warranty_score = 1.0
        scores["warranty"] = warranty_score

        # Completion time score (faster is better, but not if suspiciously fast)
        completion_times = [q.completion_days for q in leveled.quotes if q.completion_days > 0]
        if completion_times and quote.completion_days > 0:
            avg_time = sum(completion_times) / len(completion_times)
            if quote.completion_days < avg_time * 0.5:
                # Suspiciously fast
                completion_score = 0.5
            elif quote.completion_days > avg_time * 1.5:
                # Too slow
                completion_score = 0.3
            else:
                max_time = max(completion_times)
                min_time = min(completion_times)
                if max_time > min_time:
                    completion_score = 1 - ((quote.completion_days - min_time) / (max_time - min_time))
                else:
                    completion_score = 1.0
        else:
            completion_score = 0.5  # Neutral if not specified
        scores["completion_time"] = completion_score

        # Payment terms score (less advance is better for owner)
        advances = [q.mobilization_advance for q in leveled.quotes]
        min_adv = min(advances)
        max_adv = max(advances)
        if max_adv > min_adv:
            payment_score = 1 - ((quote.mobilization_advance - min_adv) / (max_adv - min_adv))
        else:
            payment_score = 1.0
        scores["payment_terms"] = payment_score

        # Scope completeness score (fewer exclusions is better)
        all_exclusions = [len(q.exclusions) for q in leveled.quotes]
        max_exc = max(all_exclusions) if all_exclusions else 0
        min_exc = min(all_exclusions) if all_exclusions else 0
        if max_exc > min_exc:
            scope_score = 1 - ((len(quote.exclusions) - min_exc) / (max_exc - min_exc))
        else:
            scope_score = 1.0
        scores["scope_completeness"] = scope_score

        # Calculate weighted total
        total_score = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)

        return total_score

    def _generate_reasons(
        self,
        recommended,
        lowest,
        leveled: LeveledComparison,
        scores: Dict,
    ) -> List[str]:
        """Generate reasons for recommendation."""
        reasons = []

        if recommended.subcontractor_name == lowest.subcontractor_name:
            reasons.append("Lowest price after leveling")
        else:
            price_diff = recommended.leveled_total - lowest.leveled_total
            price_pct = (price_diff / lowest.leveled_total) * 100
            reasons.append(f"Price is {price_pct:.1f}% higher than lowest, but offers better value")

        # Warranty comparison
        if recommended.warranty_months > 12:
            reasons.append(f"Extended warranty of {recommended.warranty_months} months")

        # Scope comparison
        if len(recommended.exclusions) == 0:
            reasons.append("Complete scope with no exclusions")
        elif len(recommended.exclusions) < len(lowest.exclusions):
            reasons.append("Fewer exclusions than lowest bidder")

        # Completion time
        if recommended.completion_days > 0:
            avg_time = sum(q.completion_days for q in leveled.quotes if q.completion_days > 0) / max(1, len([q for q in leveled.quotes if q.completion_days > 0]))
            if recommended.completion_days < avg_time:
                reasons.append(f"Faster completion ({recommended.completion_days} days vs avg {avg_time:.0f} days)")

        # Overall score
        reasons.append(f"Highest overall score: {scores[recommended.subcontractor_name]:.2f}")

        return reasons

    def _identify_risks(
        self,
        recommended,
        leveled: LeveledComparison,
    ) -> List[str]:
        """Identify risks with recommended bidder."""
        risks = []

        # Price outliers
        all_prices = [q.leveled_total for q in leveled.quotes]
        avg_price = sum(all_prices) / len(all_prices)
        if recommended.leveled_total < avg_price * 0.8:
            risks.append("Price significantly below average - verify capacity and financial stability")

        # Short completion time
        if recommended.completion_days > 0:
            completion_times = [q.completion_days for q in leveled.quotes if q.completion_days > 0]
            if completion_times:
                avg_time = sum(completion_times) / len(completion_times)
                if recommended.completion_days < avg_time * 0.7:
                    risks.append("Completion time significantly faster than others - may be unrealistic")

        # Many exclusions
        if len(recommended.exclusions) > 3:
            risks.append(f"Quote has {len(recommended.exclusions)} exclusions - verify final scope")

        # Short validity
        if recommended.validity_days < 15:
            risks.append(f"Quote validity only {recommended.validity_days} days - may need extension")

        # High advance
        if recommended.mobilization_advance > 20:
            risks.append(f"High mobilization advance ({recommended.mobilization_advance}%) - consider reducing")

        return risks

    def generate_negotiation_points(
        self,
        leveled: LeveledComparison,
        target_bidder: str,
    ) -> List[str]:
        """Generate negotiation points for a specific bidder."""
        points = []

        target = next((q for q in leveled.quotes if q.subcontractor_name == target_bidder), None)
        if not target:
            return ["Bidder not found"]

        lowest = min(leveled.quotes, key=lambda q: q.leveled_total)

        # Price negotiation
        if target.leveled_total > lowest.leveled_total:
            diff = target.leveled_total - lowest.leveled_total
            diff_pct = (diff / lowest.leveled_total) * 100
            points.append(f"Request {diff_pct:.1f}% price reduction to match lowest bidder")

        # Exclusions to include
        if target.exclusions:
            points.append(f"Request inclusion of: {', '.join(target.exclusions[:3])}")

        # Warranty improvement
        max_warranty = max(q.warranty_months for q in leveled.quotes)
        if target.warranty_months < max_warranty:
            points.append(f"Request warranty extension to {max_warranty} months")

        # Payment terms
        min_advance = min(q.mobilization_advance for q in leveled.quotes)
        if target.mobilization_advance > min_advance:
            points.append(f"Request mobilization advance reduction to {min_advance}%")

        return points
