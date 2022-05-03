import datasets

class MultiRC(datasets.Metric):
    def _info(self):

        return datasets.MetricInfo(
            description="My metric for multirc",
            citation="",
            inputs_description="",
            features=datasets.Features({
                "predictions": datasets.Value("string"),
                "references": [datasets.Value("string")]
            }),
            codebase_urls=[],
            reference_urls=[]
        )

    def _compute(self, predictions, references):
        rouge = datasets.load_metric("rouge")
        results = []
        for pred, ref in zip(predictions, references):
            p = [pred for _ in range(len(ref))]
            result = rouge.compute(predictions=p, references=ref)
            results.append(result['rougeL'].high.fmeasure)

        return float(sum(results) / len(results))