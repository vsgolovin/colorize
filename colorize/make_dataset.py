import os
from pathlib import Path
import torchvision as tv
from PIL import Image
import random
import shutil
import numpy as np
from dataset_utils import crop_resize


def main(
    root: str,
    train_size: int,
    val_size: int,
    test_size: int,
    save_to: str = ".",
    min_width: int = 224,
    min_height: int = 224,
    display_progress: bool = True,
):
    # read file names
    fnames_dir = os.path.join(root, "ImageSets", "CLS-LOC")
    train_files = read_file_names(os.path.join(fnames_dir, "train_cls.txt"))
    assert len(train_files) >= train_size
    val_files = read_file_names(os.path.join(fnames_dir, "val.txt"))
    test_files = read_file_names(os.path.join(fnames_dir, "test.txt"))

    # collect dataset
    for input_fnames, subset, n in zip(
        (train_files, val_files, test_files),
        ("train", "val", "test"),
        (train_size, val_size, test_size),
    ):
        input_files = [
            os.path.join(root, "Data", "CLS-LOC", subset, fname)
            for fname in input_fnames
        ]
        output_dir = os.path.join(save_to, subset)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if display_progress:
            print(f"Creating {subset} dataset:")
        resize = subset != "train"  # center crop + resize -> 224x224
        copy_images(
            input_files,
            output_dir,
            n,
            min_width,
            min_height,
            display_progress=display_progress,
            resize=resize,
        )


def read_file_names(txtfile: str) -> list[str]:
    suffix = ".JPEG"
    with open(txtfile) as infile:
        file_names = [line.split()[0] + suffix for line in infile]
    return file_names


def copy_images(
    input_files: list[str],
    output_dir: str,
    n: int,
    min_width: int = 224,
    min_height: int = 224,
    display_progress: bool = True,
    resize: bool = False,
):
    """
    Copy a random sample of `n` images `input_files` to `output_dir`.
    Skip images that are smaller than `(min_width, min_height)`.
    """
    idx = list(range(len(input_files)))
    random.shuffle(idx)  # inplace
    count = 0  # number of copied images
    for index in idx:
        if display_progress:
            print(f"\r{count} / {n} ({count/n * 100:.1f}%)", end="")
        if count >= n:
            break
        input_file = input_files[index]
        _, fname = os.path.split(input_file)
        with Image.open(input_file) as img:
            w, h = img.size
            if w >= min_width and h >= min_height and not is_grayscale(img):
                output_file = os.path.join(output_dir, fname)
                if not resize or (w == min_width and h == min_height):
                    shutil.copy(input_file, output_file)
                else:  # crop and resize image
                    img = crop_resize(img, min_width, min_height)
                    img.save(output_file)
                count += 1
    if display_progress:
        print()


def is_grayscale(img: Image) -> bool:
    arr = np.asarray(img)
    # check number of channels
    if arr.ndim == 2 or arr.shape[2] == 1:
        return True
    return False

def Grayscale_folder(inp_folder: str, out_folder: int, n_channels: int=3, n_pictures=1000, image_format: str="JPEG"):
    assert os.path.exists(inp_folder)
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    grayscale = tv.transforms.Grayscale(num_output_channels=n_channels)
    filenames = tuple(f 
    for f in os.listdir(inp_folder) 
    if f.endswith(image_format)
    )[:n_pictures]
    for filename in filenames:
        with Image.open(Path(inp_folder, filename)) as img:
            gray = grayscale(img)
            gray.save(Path(out_folder, filename))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a subset of kaggle ImageNet dataset"
    )
    parser.add_argument("root", type=str, help="path to ImageNet dataset")
    parser.add_argument(
        "--train-size", type=int, help="train dataset size", default=100000
    )
    parser.add_argument(
        "--val-size", type=int, help="validation dataset size", default=5000
    )
    parser.add_argument(
        "--test-size", type=int, help="test dataset size", default=10000
    )
    parser.add_argument(
        "--save-to", type=str, help="output directory", default="./data"
    )
    parser.add_argument(
        "--min-width", type=int, help="minimum image width", default=224
    )
    parser.add_argument(
        "--min-height", type=int, help="minimum image height", default=224
    )
    parser.add_argument(
        "--hide-progress",
        dest="show_progress",
        action="store_false",
        help="do not display current progress",
    )
    parser.set_defaults(show_progress=True)
    args = parser.parse_args()

    random.seed(42)
    main(
        root=args.root,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        save_to=args.save_to,
        min_width=args.min_width,
        min_height=args.min_height,
        display_progress=args.show_progress,
    )
