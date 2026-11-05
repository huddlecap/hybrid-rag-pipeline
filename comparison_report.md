=== Chunking Strategy Comparison ===

Strategy          N  Retrieval  Correctness Faithfulness   Citation
----------------------------------------------------------------------
fixed            30      96.7%        86.7%        26.7%      93.3%
structure        30     100.0%        90.0%        40.0%     100.0%
semantic         30      93.3%        83.3%        30.0%      96.7%

=== Paired Wins/Losses (questions in common) ===

fixed vs structure (common questions: 30)
  retrieval_relevance  fixed:   0  structure:   1  ties:  29
  answer_correctness   fixed:   2  structure:   3  ties:  25
  faithfulness         fixed:   3  structure:   7  ties:  20
  citation_accuracy    fixed:   0  structure:   2  ties:  28

fixed vs semantic (common questions: 30)
  retrieval_relevance  fixed:   2  semantic:   1  ties:  27
  answer_correctness   fixed:   3  semantic:   2  ties:  25
  faithfulness         fixed:   5  semantic:   6  ties:  19
  citation_accuracy    fixed:   1  semantic:   2  ties:  27

structure vs semantic (common questions: 30)
  retrieval_relevance  structure:   2  semantic:   0  ties:  28
  answer_correctness   structure:   4  semantic:   2  ties:  24
  faithfulness         structure:   8  semantic:   5  ties:  17
  citation_accuracy    structure:   1  semantic:   0  ties:  29

RESULT: Overall winner: structure (won 8/12 metric-pair matchups)
  fixed: 2 metric wins
  structure: 8 metric wins
  semantic: 2 metric wins
