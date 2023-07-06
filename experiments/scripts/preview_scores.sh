jq "{
 em: .ExactMatch.em,
 BERTScore: .BERTScore.f1_mean,
 Rouge: .ROUGE.rouge_avg_fmeasure_mean,
 Rouge1: .ROUGE.rouge1.fmeasure_mean,
 P1P2Pct: .Paper1Paper2Percentage.mean,
 QuestionPct: .QuestionPercentage.mean,
 QuestionNumberComparison: .QuestionNumberComparison,
 QuestionSnippetOverlap: .QuestionSnippetOverlap,
 LengthChange: .LengthChange,
 Sari: .Sari,
 Clarification: {
    iou_mean: .Clarification.iou_mean,
    p_mean: .Clarification.p_mean,
    r_mean: .Clarification.r_mean,
    f1_mean: .Clarification.f1_mean},
 QA: {
    qa_em: .qa_ExactMatch.em,
    qa_rouge: .qa_ROUGE.rouge_avg_fmeasure_mean},
}" $1/scores.json