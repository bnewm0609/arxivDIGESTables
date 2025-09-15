from collections import defaultdict
import json
from pathlib import Path
from typing import Optional, Any, Iterator, Sequence

from omegaconf import DictConfig
import pandas as pd
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, BatchEncoding

from paper_comparison.custom_types import Table
from papermage.magelib import Document


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line.strip()) for line in f]


def load_debug_json(path):
    with open(path) as f:
        pass

def get_intro(doc):
    intro = ""
    # Find intro section based on the headings
    if hasattr(doc, "headings") and hasattr(doc, "sections"):
        for heading in doc.headings:
            if "intro" in heading.text.lower().replace(" ", ""):
                intro_section = "\n".join([s.text for s in heading.intersect_by_span('sections')])
                intro += "\n" + intro_section
        # If there is no introduction heading, use the first few sections with no
        #   assigned headings
        if intro == "":
            for section in doc.sections[:2]:
                intro += "\n" + section.text
    else:
        # If there is no section indexed, use the first five paragraphs
        intro = "\n".join([p.text for p in doc.paragraphs[:5]])
    return intro


class Dataset:
    """Super class that contains logic for datasets.

    Attributes:
        args: Contains configuration material for the run.
        split: Which split this data belongs to (one of "train", "val" (validation), or "test").
        tokenizer: Tokenizer (usually from HuggingFace). Used for both API models and local models.
        data: List of dictionaries of samples that make up the dataset.
        x_label: The key of the dictionary of each that represents the input.
        y_label: The key of the dictionary of each sample that contains the gold output.
    """

    def __init__(self, args: DictConfig, split: str, tokenizer: Optional[PreTrainedTokenizer]) -> None:
        """Initialize a dataset by reading data and creating the dataloader.

        Args:
            args: The configuration for the experiment.
            tokenizer: Tokenizer (usually from HuggingFace) for tokenizing inputs.
            split: Which split this data belongs to (one of "train", "val" (validation), or "test").
        """

        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.data = self.read_data(args.data.get(split).path)

        # default keys for in the data list dicts
        self.x_label = "x"
        self.y_label = "y"

        self.create_data_loader()

    def create_data_loader(self):
        """Bundle `self.data` into a pytorch DataLoader.

        Only shuffles the training set.
        """

        self.dataloader = DataLoader(
            self.data,
            batch_size=self.args.model.batch_size,
            collate_fn=self.collate,
            shuffle=self.split == "train",
        )

    def read_data(self, papers_path: str, tables_path: Optional[str]) -> Sequence[Any]:
        """Read data from path and returns it as a Sequence that can be loaded into a torch dataloader.

        Subclasses will override this method.

        Args:
            data_path_or_name: The path to the dataset (or name if using a huggingface dataset).

        Returns:
            Sequence (usually a list) containing the data.
        """

        raise NotImplementedError

    def collate(self, batch: Sequence[Any]) -> BatchEncoding:
        """Tokenize the batch and convert it to tensors.

        Collate examples from the batch for input to the model in the torch DataLoader.

        Args:
            batch: The samples that constitue a single batch.

        Returns:
            The collated batch for input into a Huggingface model.
        """

        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        """Return the example at the given index from the dataset."""
        return self.data[idx]

    def __iter__(self) -> Iterator[Any]:
        """Return an iterator for iterating over the dataset."""
        return iter(self.data)


# class HuggingfaceDataset(Dataset):
#     """The default dataset for HuggingFace models.

#     Data is read from a `jsonl` file, tokenized and batched using a Huggingface transformers tokenizer."""

#     def read_data(self, data_path_or_name: str) -> Sequence[Any]:
#         """Read data from `jsonl` file.

#         Args:
#             data_path_or_name: The path to the json lines file containing the dataset.

#         Returns:
#             list[dict] containing the data.
#         """

#         data = []
#         with open(data_path_or_name) as f:
#             for line in f:
#                 sample = json.loads(line.strip())
#                 data.append(sample)
#         return data

#     def collate(self, batch: Sequence[Any]) -> BatchEncoding:
#         """Tokenize each batch using the huggingface tokenizer.

#         See overridden function in Dataset for information on args and return type.
#         """

#         inputs, targets = zip(*[(sample[self.x_label], sample[self.y_label]) for sample in batch])

#         inputs = self.tokenizer(list(inputs), padding=True, truncation=True)
#         with self.tokenizer.as_target_tokenizer():
#             targets = self.tokenizer(list(targets), padding=True, truncation=True)

#         # Replace all "pad token ids" with -100 because we want to ignore them when calculating the loss
#         targets["input_ids"] = [
#             [(token_id if token_id != self.tokenizer.pad_token_id else -100) for token_id in label]
#             for label in targets["input_ids"]
#         ]

#         inputs["labels"] = targets["input_ids"]
#         return inputs.convert_to_tensors(tensor_type="pt")


# class TemplateDataset(HuggingfaceDataset):
#     """A dataset for templating prompts for API endpoints.

#     This dataset was designed with the OpenAI chat and completion endpoints in mind, but also
#     can work with other API providers.
#     This dataset probably should not be
#     Could also be used for completition endpoint if so desired.
#     Probably shouldn't be used for fewshot learning. Look at `models.FewShotModel`
#     for the code for doing few shot learning.

#     Updates self.data directly because we send text to GPT3.

#     Attributes:
#         template: The template to format the data in.
#     """

#     def __init__(self, args: DictConfig, tokenizer: PreTrainedTokenizer, split: str) -> None:
#         """Initialize the dataset.

#         In addition to loading the data, also load the template. Also, save the template and a sample example
#         in the results directory for debugging the template. Finally, try to create the dataloader, but it
#         is not necessary because we are not doing batching. The `train.jsonl` might not exist, so skip this
#         step if it doesn't.
#         """

#         # set up the dataset
#         self.args = args
#         self.split = split
#         self.tokenizer = tokenizer

#         self.template = self.load_template(args.data.template)
#         self.data = self.read_data(args.data.get(split).path, self.template)

#         # it's good to have an example data sample for debugging and reproducibility
#         # so save one in the results dir:
#         if self.split == "val":
#             save_dir = Path(args.results_path) / (args.data.val._id + "-" + args.generation._id)
#             save_dir.mkdir(parents=True, exist_ok=True)

#             with open(save_dir / "sample_val_data.json", "w") as f:
#                 json.dump(self.data[0], f, default=dict, ensure_ascii=False)

#             # additionally, save the template:
#             with open(Path(args.results_path) / "template.json", "w") as f:
#                 if isinstance(self.template, str):
#                     f.write(self.template)
#                 else:
#                     template_dict = [t.dict() for t in self.template]
#                     json.dump(template_dict, f)

#         # default keys for in the data list dicts
#         self.x_label = "x"
#         self.y_label = "y"

#         try:
#             self.create_data_loader()
#         except ValueError:
#             if split != "train":
#                 raise ValueError
#             else:
#                 print(f"No data found! File {args.data.get(split).path} is empty.")

#     def load_template(self, template: Union[str, list]) -> Union[list[OpenAIChatMessage], str]:
#         """Load the template from the config.

#         The passed template takes one of three forms:
#         1. a list[dict[str, str]] (for the OpenAI Chat Endpoint). The keys are the role ("user", "system")
#             and the values are the template for that message. The dict[str, str] is converted into an
#             `OpenAIChatMessage`.
#         2. a string containing the template (for the OpenAI Completion or Claude endpoints)
#         3. a string with a yaml filepath to either of the two above template types.
#         The template strings are jinja templates.

#         Args:
#             template (Union[str, list]): The template or a path to a yaml file with template.

#         Returns:
#             list[OpenAIChatMessage] for the OpenAI Chat API case and a str for the Completion or Claude
#             cases with the template. The template is not filled at this point.
#         """

#         # there are a few choices for template:
#         if isinstance(template, str) and template.startswith("https://"):
#             # assume that the template is a public google sheet and read it into pandas as a csv
#             raise NotImplementedError()
#         elif isinstance(template, str) and len(template) < 256 and Path(template).is_file():
#             # read the template from the file path
#             with open(template) as f:
#                 if template.endswith("yaml"):
#                     template = yaml.safe_load(f)["template"]
#                     if isinstance(template, list):
#                         template = [OpenAIChatMessage(**item) for item in template]
#                 else:
#                     template = f.read()
#         elif isinstance(template, str):
#             # assume the template is for a non-chat model
#             template = template
#         elif isinstance(template, list) or isinstance(template, ListConfig):
#             # assume that the passsed thing is the template dict itself
#             template = [OpenAIChatMessage(**item) for item in template]
#         else:
#             raise ValueError("Template must be either a list, url or path to a valid file")

#         return template

#     def read_data(  # type: ignore
#         self, data_path_or_name: str, template: Union[str, list[OpenAIChatMessage]]  # type: ignore
#     ) -> Sequence[Any]:  # type: ignore
#         """Read the data by filling in the template.

#         Args:
#             data_path_or_name (str): path to the `jsonl` file containing the data.
#             template (Union[str, list[OpenAIChatMessage]]): the jinja template that will be filled in.

#         Returns:
#             A list of samples, where the value at `self.data[i][self.x_label]` is the template filled in with
#             the data for the `i`th sample.
#         """

#         data = []
#         with open(data_path_or_name) as f:
#             for line in f:
#                 sample = json.loads(line.strip())
#                 # overwrite full_text if it's present:
#                 if "full_text" in sample:
#                     sample["full_text"] = QasperFullText(
#                         title=sample["title"],
#                         abstract=sample["abstract"],
#                         full_text=[QasperSection(**section) for section in sample["full_text"]["full_text"]],
#                     )

#                     while (
#                         len(self.tokenizer.tokenize(str(sample["full_text"]))) > 8080 - 100
#                     ):  # 755:  # len(prompt)
#                         if not sample["full_text"].full_text[-1].paragraphs:
#                             sample["full_text"].full_text.pop(-1)
#                         sample["full_text"].full_text[-1].paragraphs.pop(-1)

#                 # substitute any variables into the template
#                 if isinstance(template, str):
#                     new_messages = Template(template, undefined=DebugUndefined).render(sample)
#                 else:
#                     new_messages = []
#                     for chat_message in template:
#                         new_message_content = Template(chat_message.content, undefined=DebugUndefined).render(
#                             sample
#                         )  # any extra elements will be ignored
#                         new_messages.append(
#                             OpenAIChatMessage(
#                                 role=chat_message.role,
#                                 content=new_message_content,
#                             )
#                         )
#                 data.append(
#                     {
#                         "idx": sample["idx"],
#                         "x": new_messages,
#                         "y": sample["y"],
#                     }
#                 )
#         if self.split == "train":
#             # save a few training examples with the prompts for debugging
#             with open(Path(self.args.results_path) / "filled_train_templates.jsonl", "w") as f:
#                 for sample in data:
#                     try:
#                         f.write(json.dumps(sample) + "\n")
#                     except TypeError:
#                         f.write(
#                             json.dumps(
#                                 {
#                                     "idx": sample["idx"],
#                                     "x": [json.loads(message.json()) for message in sample["x"]],
#                                     "y": sample["y"],
#                                 }
#                             )
#                         )

#             print(
#                 "Saved train prompts to:",
#                 Path(self.args.results_path) / "filled_train_templates.jsonl",
#             )
#         return data

#     def collate(self, batch: Sequence[Any]) -> BatchEncoding:
#         """Use the huggingface tokenizer to tokenize each batch"""
#         inputs, targets = zip(*[(sample[self.x_label], sample[self.y_label]) for sample in batch])

#         inputs = self.tokenizer(inputs, padding=True, truncation=True, max_length=1023, return_tensors="pt")
#         inputs["labels"] = torch.clone(inputs["input_ids"])
#         token_type_ids = inputs.pop("token_type_ids")
#         inputs["labels"][token_type_ids == 0] = -100  # we don't want to caculate loss on the input tokens

#         return inputs


class DebugAbstractsDataset(Dataset):
    def __init__(self, args: DictConfig, split: str, tokenizer=None) -> None:
        self.args = args
        self.split = split

        # default keys for in the data list dicts
        self.x_label = "x"
        self.y_label = "y"

        # self.tokenizer = tokenizer
        # self.data = self.read_data(args.data.get(split).path)
        self.data = self.read_data(args.data.papers_path, args.data.tables_path, args.data.dataset_path)

    def read_data(self, papers_path: str, tables_path: Optional[str], dataset_path: Optional[str]) -> Sequence[Any]:
        # load papers
        papers = load_jsonl(papers_path)

        # load datasets
        datasets = load_jsonl(dataset_path)
        # load tables (if they exist)
        tabid_to_table = {}
        if tables_path is not None and Path(tables_path).exists():
            tables_json = load_jsonl(tables_path)
            for table_json in tables_json:
                schema = set(table_json["table"].keys())
                # find the data that has the same tabid but key is "_table_hash"
                for dataset in datasets:
                    if dataset["_table_hash"] == table_json["tabid"]:
                        caption = dataset["caption"]
                        ics_papers = dataset["ics_papers"]
                        ics_captions = dataset["ics_caption"]
                        in_text_refs = dataset["in_text_ref"]
                        break
                table = Table(tabid=table_json["tabid"], schema=schema, values=table_json["table"], caption=caption if caption else None,
                              icscaption=ics_captions if ics_captions else None, icspaper=ics_papers if ics_papers else None, intextref=in_text_refs if in_text_refs else None)
                tabid_to_table[table_json["tabid"]] = table
                ####### NEWLY ADDED #######
                # add "ics_papers", "ics_caption", "in_text_ref" to the tabid_to_table
                # tabid_to_table["ics_caption"] = ics_captions
                # tabid_to_table["ics_papers"] = ics_papers
                # tabid_to_table["in_text_ref"] = in_text_refs

        # match table ids to papers:
        tabid_to_paper = defaultdict(list)
        for paper in papers:
            tabids = paper.pop("tabids")
            for tabid in tabids:
                tabid_to_paper[tabid].append(paper)

        data = []
        for tabid in tabid_to_paper:
            if tabid_to_table.get(tabid) is None:
                continue
            data.append(
                {"idx": tabid, self.x_label: tabid_to_paper[tabid], self.y_label: tabid_to_table.get(tabid)}
            )

        return data


class FullTextsDataset(Dataset):
    def __init__(self, args: DictConfig, split: str, tokenizer=None) -> None:
        self.args = args
        self.split = split

        # default keys for in the data list dicts
        self.x_label = "x"
        self.y_label = "y"

        # self.tokenizer = tokenizer
        # self.data = self.read_data(args.data.get(split).path)
        self.data = self.read_data(args.data.papers_path, args.data.tables_path)

    def read_data(self, papers_path: str, tables_path: Optional[str]) -> Sequence[Any]:
        # load papers
        papers = load_jsonl(papers_path)
        with open(self.args.data.full_texts_path) as f:
            ft_raws = list(f)

        # load/create full text index
        full_texts_index = {}
        full_texts_index_path = Path(self.args.results_path) / "full_texts_index.json"
        if full_texts_index_path.exists():
            with open(full_texts_index_path) as f:
                full_texts_index = json.load(f)
        else:
            for idx, _ft in enumerate(ft_raws):
                ft = json.loads(_ft)
                full_texts_index[ft['metadata']['corpusId']] = idx

            with open(full_texts_index_path, "w") as f:
                json.dump(full_texts_index, f)
        for paper in papers:
            corpus_id = str(paper.get("corpus_id"))
            if corpus_id is not None and corpus_id in full_texts_index:
                doc = Document.from_json(json.loads(ft_raws[full_texts_index[corpus_id]]))
                paper["full_text"] = doc.symbols
                paper["intro"] = get_intro(doc)

        # load tables (if they exist)
        tabid_to_table = {}
        if tables_path is not None and Path(tables_path).exists():
            tables_json = load_jsonl(tables_path)
            for table_json in tables_json:
                table_df = pd.DataFrame(table_json["table"])
                schema = set(table_df.columns)
                table = Table(
                    tabid=table_json["tabid"], schema=schema, values=table_df.to_dict(), dataframe=table_df
                )
                tabid_to_table[table_json["tabid"]] = table

        # match table ids to papers:
        tabid_to_paper = defaultdict(list)
        for paper in papers:
            tabids = paper.pop("tabids")
            for tabid in tabids:
                tabid_to_paper[tabid].append(paper)

        data = []
        for tabid in tabid_to_paper:
            data.append(
                {"idx": tabid, self.x_label: tabid_to_paper[tabid], self.y_label: tabid_to_table.get(tabid)}
            )

        return data


def load_data(args: DictConfig):
    if args.data.type == "debug_abstracts":
        return DebugAbstractsDataset(args, "val")
    elif args.data.type == "full_texts":
        return FullTextsDataset(args, "val")
