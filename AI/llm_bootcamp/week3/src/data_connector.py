from datasets import load_dataset, DatasetDict


class DataConnector:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_data(
        data_path: str,
        dataset_name: str | None = None,
        split_list: list[str] = ["train", "validation"],
        split_perc: int | None = None
    ) -> DatasetDict:
        if split_perc is not None:
            SPLIT_ERROR = "Data split percentage should between 1-100."
            assert 0 < split_perc <= 100, SPLIT_ERROR
            split_conf = {k: split_perc for k in split_list}
            dataset = load_dataset(
                path=data_path,
                name=dataset_name,
                split=[f"{k}[:{split_conf[k]}%]" for k in split_list]
            )
            return DatasetDict({
                k: v for k, v in zip(split_conf.keys(), dataset)
            })
        return load_dataset(
            path=data_path,
            name=dataset_name,
            split=split_list
        )
