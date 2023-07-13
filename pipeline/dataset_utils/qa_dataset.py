import base64
from io import BytesIO
import re
import json
import contextlib
import os

from PIL import ImageFile
from torchvision import transforms
from torch.utils.data import Dataset

from pipeline.multi_instruct_data_utils.transforms import *
from pipeline.multi_instruct_data_utils.multi_instruct_dataset import collate_fn


label_map = {"entailment": 0, "not_entailment": 1}

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

FLAMINGO_MEAN = [0.481, 0.458, 0.408]
FLAMINGO_STD = [0.269, 0.261, 0.276]

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class QADataset(Dataset):
    def __init__(self, args, is_test=False):
        # Input parameters.
        self.args = args
        self.task_name = args.task
        self.is_test = is_test
        self.tokenizer = args.tokenizer

        self.max_src_length = args.max_src_length
        self.max_tgt_length = args.max_tgt_length

        self.seed = args.pretrain_seed
        self.code_dict_size = args.code_dict_size
        self.patch_image_size = args.patch_image_size
        self.code_image_size = args.code_image_size

        self.epoch = 0

        scales = [(args.patch_image_size, args.patch_image_size)]

        self.patch_resize_transform = transforms.Compose(
            [
                RandomResize(scales),
                transforms.CenterCrop(args.patch_image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=FLAMINGO_MEAN, std=FLAMINGO_STD),
            ]
        )

        self.file_path = args.multi_instruct_path
        self.data_root = args.data_root


        assert os.path.exists(
            self.file_path
        ), "Error: The local datafile {} not exists!".format(self.file_path)
        self.separator = "\t"

        self.dataset = json.load(open(self.file_path, "r"))

        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id])
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id])
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])
        self.rank = args.rank


    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def process_image_text_pair(self, index):
        uniq_id = self.dataset[index]["image_id"]
        image_path = os.path.join(self.data_root, self.dataset[index]["image"])
        image = Image.open(image_path).convert("RGB")
        patch_image = (self.patch_resize_transform(image) if type != "positioning" else None)
        question = self.dataset[index]["input"]
        max_tgt_length = 256
        answer = self.dataset[index]["output"][:max_tgt_length]

        src_text = self.tokenizer(
            f"<image>User: {question} GPT:<answer> {answer}<|endofchunk|>",
            return_tensors="pt",
            add_special_tokens=False,
        )
        src_item = src_text["input_ids"].squeeze(0)
        src_item_mask = src_text["attention_mask"].squeeze(0)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        src_item_mask = torch.cat([self.bos_mask, src_item_mask, self.eos_mask])

        example = {
            "id": uniq_id,
            "source": src_item,
            "text_mask": src_item_mask,
            "patch_image": patch_image,
        }

        examples = [example]

        return examples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        with numpy_seed(self.seed, self.epoch):
            pair_samples = self.process_image_text_pair(index)
            if pair_samples is None:
                return self.__getitem__(index + 1)
        return pair_samples

    def collate(self, samples):
        samples_v1 = []  # containing image-text pairs
        for sample_tuple in samples:
            samples_v1.append(sample_tuple[0])

        res_v1 = collate_fn(
            samples_v1,
            pad_idx=self.tokenizer.pad_token_id,
            eos_idx=self.tokenizer.eos_token_id,
        )
        return res_v1
