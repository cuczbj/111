import os
import shutil
from pathlib import Path
import gdown
import argparse


def wgetgdrive(file_id, output_path):
    URL = f"https://docs.google.com/uc?export=download&id={file_id}"
    gdown.download(URL, str(output_path), quiet=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=str, default="./data")
    parser.add_argument("--data-key", type=str, default="rel3d")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.data_key == "rel3d":
        # Ackownledgement: Goyal, A., Yang, K., Yang, D., & Deng, J. (2020). Rel3d: A minimally contrastive benchmark for grounding spatial relations in 3d. Advances in Neural Information Processing Systems, 33, 10514-10525.
        file_id = "1sebXU7pZ0FI7lG28OkH5qSnWHpADQRFi"
        wgetgdrive(file_id, tmp_dir / "data_min.zip")
        os.system(f"unzip -o {tmp_dir / 'data_min.zip'}")
        shutil.move("data_min", target_dir / "test_rel3d")
        os.remove(tmp_dir / "data_min.zip")

    elif args.data_key == "spatialsense+":
        import zipfile
        import tarfile
        
        spatialsense_dir = target_dir / "spatialsense"
        spatialsense_dir.mkdir(parents=True, exist_ok=True)
        spatialsense_image_dir = spatialsense_dir / "images"
        spatialsense_image_dir.mkdir(parents=True, exist_ok=True)
        
        zip_path = tmp_dir / 'spatialsense.zip'
        
        # 检查文件是否存在
        if not zip_path.exists():
            print(f"错误：请将 spatialsense.zip 放到 {zip_path}")
            return
        
        # 解压 ZIP 文件
        print("正在解压 spatialsense.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(spatialsense_dir)
        
        # 解压 tar.gz 文件
        tar_path = spatialsense_dir / 'images.tar.gz'
        if tar_path.exists():
            print("正在解压 images.tar.gz...")
            with tarfile.open(tar_path, 'r:gz') as tar_ref:
                tar_ref.extractall(spatialsense_image_dir)
        
        # 检查或下载注释文件
        json_path = spatialsense_dir / "annots_spatialsenseplus.json"
        if not json_path.exists():
            print("正在下载 SpatialSense+ 注释文件...")
            file_id = "1vIOozqk3OlxkxZgL356pD1EAGt06ZwM4"
            wgetgdrive(file_id, json_path)
        else:
            print("找到已存在的注释文件")
        
        # 清理临时文件
        if zip_path.exists():
            os.remove(zip_path)
        if tar_path.exists():
            os.remove(tar_path)


    elif args.data_key == "ibot":
        # Acknowledgement: Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., & Kong, T. ibot: Image BERT Pre-training with Online Tokenizer. In International Conference on Learning Representations.
        file_id = "1nO06i4xc8RAp2W8xm06cO0oTHwIZ2VAX"
        pretrained_ckp_dir = target_dir / "pretrained_checkpoints"
        pretrained_ckp_dir.mkdir(parents=True, exist_ok=True)
        wgetgdrive(file_id, pretrained_ckp_dir / "ibot_vit_base_patch16.pth")

    else:
        raise ValueError(f"Unknown data key: {args.data_key}")

    tmp_dir.rmdir()


if __name__ == "__main__":
    main()
