from datasets import load_dataset, DatasetDict


class DataConnector:
    
    # TODO:  Implement the DataConnector to pull the data from the path. 
    # As an example, you can choose to pull the data from the datasets 
    # package with the load_dataset function.

    @staticmethod
    def get_data(
        data_path: str,
        dataset_name: str | None = None,
        split_list: list[str] = ["train", "validation"],
        split_perc: int | None = None
    ) -> DatasetDict:
        if split_perc is not None:
            SPLIT_ERROR = "Data split percentage should between (0, 100]."
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
        return DatasetDict(
            {
                k: v for k, v in zip(
                    split_list,
                    load_dataset(
                        path=data_path,
                        name=dataset_name,
                        split=split_list
                    )
                )
            }
        )

