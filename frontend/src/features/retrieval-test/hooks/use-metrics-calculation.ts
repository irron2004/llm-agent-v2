import { useMemo } from "react";
import {
  SearchResult,
  ResultMetrics,
  AggregatedMetrics,
  RetrievalTestResult,
} from "../types";

export function calculateMetrics(
  searchResults: SearchResult[],
  groundTruthDocIds: string[]
): ResultMetrics {
  const relevantRanks: number[] = [];

  searchResults.forEach((result, index) => {
    if (groundTruthDocIds.includes(result.id)) {
      relevantRanks.push(index + 1);
    }
  });

  const firstRelevantRank =
    relevantRanks.length > 0 ? Math.min(...relevantRanks) : null;

  const hit_at_1 = relevantRanks.some((rank) => rank <= 1);
  const hit_at_3 = relevantRanks.some((rank) => rank <= 3);
  const hit_at_5 = relevantRanks.some((rank) => rank <= 5);
  const hit_at_10 = relevantRanks.some((rank) => rank <= 10);

  const reciprocal_rank = firstRelevantRank ? 1.0 / firstRelevantRank : null;

  return {
    hit_at_1,
    hit_at_3,
    hit_at_5,
    hit_at_10,
    reciprocal_rank,
    first_relevant_rank: firstRelevantRank,
  };
}

export function aggregateMetrics(
  results: RetrievalTestResult[]
): AggregatedMetrics {
  const total = results.length;

  if (total === 0) {
    return {
      total_queries: 0,
      hit_at_1: 0,
      hit_at_3: 0,
      hit_at_5: 0,
      hit_at_10: 0,
      mrr: 0,
      avg_first_relevant_rank: null,
    };
  }

  let hitAt1Count = 0;
  let hitAt3Count = 0;
  let hitAt5Count = 0;
  let hitAt10Count = 0;
  let sumReciprocalRank = 0;
  let sumFirstRelevantRank = 0;
  let countWithRelevant = 0;

  results.forEach((result) => {
    const m = result.metrics;
    if (m.hit_at_1) hitAt1Count++;
    if (m.hit_at_3) hitAt3Count++;
    if (m.hit_at_5) hitAt5Count++;
    if (m.hit_at_10) hitAt10Count++;

    if (m.reciprocal_rank !== null) {
      sumReciprocalRank += m.reciprocal_rank;
    }

    if (m.first_relevant_rank !== null) {
      sumFirstRelevantRank += m.first_relevant_rank;
      countWithRelevant++;
    }
  });

  return {
    total_queries: total,
    hit_at_1: hitAt1Count / total,
    hit_at_3: hitAt3Count / total,
    hit_at_5: hitAt5Count / total,
    hit_at_10: hitAt10Count / total,
    mrr: sumReciprocalRank / total,
    avg_first_relevant_rank:
      countWithRelevant > 0 ? sumFirstRelevantRank / countWithRelevant : null,
  };
}

export function useMetricsCalculation(results: RetrievalTestResult[]) {
  const aggregated = useMemo(() => aggregateMetrics(results), [results]);

  return { aggregated };
}
