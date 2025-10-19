#!/usr/bin/env python3
"""
合并两个版本的waymo数据集
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import ipdb

st = ipdb.set_trace


def analyze_dataset_structure(base_path: Path) -> Tuple[Set[str], int, int, int, int]:
    """
    分析数据集结构
    返回: (有meta_info.json的frames, json文件数量, 总frame文件夹数量, 没有json的frame文件夹数量, ground_info为空的json数量)
    """
    frames_with_meta = set()
    all_frames = set()
    json_count = 0
    empty_ground_info_count = 0

    if not base_path.exists():
        print(f"警告: 路径不存在 {base_path}")
        return frames_with_meta, json_count, 0, 0, 0

    # 查找所有meta_info.json文件
    for meta_file in base_path.rglob("meta_info.json"):
        json_count += 1
        # 获取相对路径 (sequence/frame)
        rel_path = meta_file.relative_to(base_path)
        # 去掉meta_info.json，保留sequence/frame路径
        frame_path = str(rel_path.parent)
        frames_with_meta.add(frame_path)

        # 检查ground_info是否为空
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                ground_info = data.get("ground_info", [])
                if not ground_info or len(ground_info) == 0:
                    empty_ground_info_count += 1
        except Exception as e:
            print(f"    警告: 读取文件出错 {meta_file}: {e}")

    # 查找所有可能的frame文件夹（两层深度：sequence/frame）
    for sequence_dir in base_path.iterdir():
        if sequence_dir.is_dir() and not sequence_dir.name.startswith("."):
            for frame_dir in sequence_dir.iterdir():
                if frame_dir.is_dir() and not frame_dir.name.startswith("."):
                    frame_rel_path = f"{sequence_dir.name}/{frame_dir.name}"
                    all_frames.add(frame_rel_path)

    frames_without_meta = all_frames - frames_with_meta

    return (
        frames_with_meta,
        json_count,
        len(all_frames),
        len(frames_without_meta),
        empty_ground_info_count,
    )


def should_filter_object(obj2: Dict) -> bool:
    """
    判断是否应该过滤掉这个object
    """
    description_ch = obj2.get("description_ch", "")

    # 检查是否包含过滤关键词
    if "指定的边界框内没有可描述的物体" in description_ch:
        return True
    if "!!!!!!!" in description_ch:
        return True

    return False


def count_filterable_objects(base_path: Path, frame_paths: Set[str]) -> Tuple[int, int]:
    """
    统计第二个版本中会被过滤的objects数量和包含被过滤objects的json文件数量
    返回: (被过滤的objects数量, 包含被过滤objects的json文件数量)
    """
    filterable_count = 0
    filter_json = 0

    for frame_path in frame_paths:
        json_path = base_path / frame_path / "meta_info.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    ground_info = data.get("ground_info", [])
                    has_filterable = False
                    for obj in ground_info:
                        if should_filter_object(obj):
                            filterable_count += 1
                            has_filterable = True
                    if has_filterable:
                        filter_json += 1
            except Exception:
                pass

    return filterable_count, filter_json


def compare_datasets(
    version1_path: Path, version2_path: Path
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    比较两个版本的数据集，返回(共同的, 只在v1的, 只在v2的)
    """
    print("正在扫描第一个版本的数据...")
    (
        v1_frames,
        v1_json_count,
        v1_total_frames,
        v1_frames_no_json,
        v1_empty_ground_info,
    ) = analyze_dataset_structure(version1_path)
    print(f"第一个版本统计:")
    print(f"  - JSON文件数量: {v1_json_count}")
    print(f"  - Frame文件夹总数: {v1_total_frames}")
    print(f"  - 有meta_info.json的frame: {len(v1_frames)}")
    print(f"  - 没有meta_info.json的frame: {v1_frames_no_json}")
    print(f"  - ground_info为空的JSON文件数: {v1_empty_ground_info}")

    print("\n正在扫描第二个版本的数据...")
    (
        v2_frames,
        v2_json_count,
        v2_total_frames,
        v2_frames_no_json,
        v2_empty_ground_info,
    ) = analyze_dataset_structure(version2_path)
    print(f"第二个版本统计:")
    print(f"  - JSON文件数量: {v2_json_count}")
    print(f"  - Frame文件夹总数: {v2_total_frames}")
    print(f"  - 有meta_info.json的frame: {len(v2_frames)}")
    print(f"  - 没有meta_info.json的frame: {v2_frames_no_json}")
    print(f"  - ground_info为空的JSON文件数: {v2_empty_ground_info}")

    common = v1_frames & v2_frames
    only_v1 = v1_frames - v2_frames
    only_v2 = v2_frames - v1_frames

    print(f"\n对比结果:")
    print(f"  - 共同的frame数: {len(common)}")
    print(f"  - 只在第一个版本的frame数: {len(only_v1)}")
    print(f"  - 只在第二个版本的frame数: {len(only_v2)}")

    # 统计第二个版本中会被过滤的objects
    if common:
        print(f"\n正在统计第二个版本中会被过滤的objects...")
        filterable_count, filter_json_count = count_filterable_objects(
            version2_path, common
        )
        print(f"  - 第二个版本中会被过滤的objects数量: {filterable_count}")
        print(f"  - 包含被过滤objects的json文件数量: {filter_json_count}")
        print(f"    (description_ch包含'指定的边界框内没有可描述的物体'或'!!!!!!!')")

    if only_v1:
        print(f"\n只在第一个版本的前10个frame:")
        for frame in list(sorted(only_v1))[:10]:
            print(f"  - {version1_path / frame}")

    if only_v2:
        print(f"\n只在第二个版本的前10个frame:")
        for frame in list(sorted(only_v2))[:10]:
            print(f"  - {version2_path / frame}")

    return common, only_v1, only_v2


def match_objects_by_bbox(
    json1_objects: List[Dict], json2_objects: List[Dict]
) -> List[Tuple[Dict, Dict]]:
    """
    根据bbox_3d匹配两个版本中的object
    返回匹配的object pairs列表
    """
    matched_pairs = []
    unmatched_j2 = list(json2_objects)

    for obj1 in json1_objects:
        bbox1 = obj1.get("bbox_3d", [])
        if not bbox1:
            continue

        # 在json2中寻找相同的bbox_3d
        best_match = None
        best_match_idx = -1

        for idx, obj2 in enumerate(unmatched_j2):
            bbox2 = obj2.get("bbox_3d", [])
            if bbox2 == bbox1:
                best_match = obj2
                best_match_idx = idx
                break

        if best_match is not None:
            matched_pairs.append((obj1, best_match))
            unmatched_j2.pop(best_match_idx)
        # else:
            # print(f"  警告: json1中的object未找到匹配，bbox_3d={bbox1[:3]}...")
            # ipdb.set_trace()

    if unmatched_j2:
        print(f"  警告: json2中有 {len(unmatched_j2)} 个object未匹配")
        # ipdb.set_trace()

    return matched_pairs, unmatched_j2


def merge_json_files(json1_path: Path, json2_path: Path, dataset_type: str) -> tuple[Dict, int, bool]:
    """
    合并两个json文件
    返回: (合并后的json, 被过滤的object数量, json2中是否有未匹配的object)
    dataset_type: "waymo", "quad", "drone"
    """
    with open(json1_path, "r", encoding="utf-8") as f:
        json1 = json.load(f)

    with open(json2_path, "r", encoding="utf-8") as f:
        json2 = json.load(f)

    # 创建json3
    json3 = {}

    # 基本信息从json1获取
    json3["sequence_name"] = json1.get("sequence_name", "")
    json3["image_path"] = json1.get("image_path", "")
    json3["lidar_path_proj"] = json1.get("lidar_path_proj", "")

    # 合并ground_info
    json1_ground_info = json1.get("ground_info", [])
    json2_ground_info = json2.get("ground_info", [])

    # 匹配objects
    matched_pairs, unmatched_j2 = match_objects_by_bbox(json1_ground_info, json2_ground_info)
    
    has_unmatched = len(unmatched_j2) > 0
    if unmatched_j2:
        # print(f"  警告: json2中有 {len(unmatched_j2)} 个object未匹配")
        print(f"\033[91m{json1_path}\033[0m")
        print(f"\033[91m{json2_path}\033[0m")
        # ipdb.set_trace()

    json3["ground_info"] = []
    filtered_count = 0

    for obj1, obj2 in matched_pairs:
        # 检查是否应该过滤掉这个object
        if should_filter_object(obj2):
            filtered_count += 1
            continue

        # 检查bbox_3d是否相同
        bbox1 = obj1.get("bbox_3d", [])
        bbox2 = obj2.get("bbox_3d", [])

        if bbox1 != bbox2:
            print(f"  警告: bbox_3d不匹配!")
            print(f"    json1: {json1_path}")
            print(f"    json2: {json2_path}")
            raise ValueError("bbox_3d不匹配!")

        
        merged_obj = {
            "class": obj1["class"].lower(),
            "caption": obj2["summary_en"],
            "caption_zh": obj2["summary_ch"],
            "attri": obj2["description_en"],
            "attri_zh": obj2["description_ch"],
            "bbox_3d": obj1["bbox_3d"],
            "bbox_2d_proj": obj1["bbox_2d_proj"],
        }
        
        # 处理others字段
        if dataset_type in ["quad", "drone"]:
            # quad/drone格式: others在根级别，需要移到ground_info里
            root_others = json1.get("others", [])
            if root_others:
                merged_obj["others"] = root_others
                merged_obj["others_num"] = len(root_others)
        else:
            # waymo格式: others在每个ground_info项里
            obj_others = obj1.get("others", [])
            if obj_others:
                merged_obj["others"] = obj_others
                merged_obj["others_num"] = len(obj_others)

        json3["ground_info"].append(merged_obj)

    # 在ground_info后面添加这些字段
    try:
        json3["image_extrinsic"] = json1["image_extrinsic"]
    except:
        json3["image_extrinsic"] = json1["extristric"]
    
    # json3["camera_distortion"] = json1.get("camera_distortion", [])
    json3["image_intrinsic"] = json1["image_intrinsic"]
    json3["pose"] = json1["pose"]
    json3["timestamp"] = json1.get("timestamp", 0)
    return json3, filtered_count, has_unmatched


def copy_files_for_frame(
    frame_rel_path: str, v1_base: Path, v2_base: Path, merge_base: Path, dataset_type: str
) -> tuple[bool, int, bool]:
    """
    复制一个frame的所有文件
    返回: (是否成功, 被过滤的object数量, json2中是否有未匹配的object)
    """
    v1_frame_dir = v1_base / frame_rel_path
    v2_frame_dir = v2_base / frame_rel_path
    merge_frame_dir = merge_base / frame_rel_path

    try:
        # 先检查JSON文件是否存在
        json1_path = v1_frame_dir / "meta_info.json"
        json2_path = v2_frame_dir / "meta_info.json"

        if not json1_path.exists() or not json2_path.exists():
            print(f"  跳过: meta_info.json不存在于 {merge_frame_dir}")
            return False, 0, False

        # 合并JSON文件
        json3, filtered_count, has_unmatched = merge_json_files(json1_path, json2_path, dataset_type)

        # 如果有未匹配的object，跳过合并
        if has_unmatched:
            # print(f"  跳过: json2中有未匹配的object {merge_frame_dir}")
            return False, filtered_count, has_unmatched

        # 检查ground_info是否为空
        if len(json3["ground_info"]) == 0:
            # print(f"  跳过: ground_info为空 {merge_frame_dir}")
            return False, filtered_count, has_unmatched

        # 只有在ground_info不为空时才创建目录并复制文件
        merge_frame_dir.mkdir(parents=True, exist_ok=True)

        # 复制第一个版本的.jpg和.npy文件
        for ext in ["*.jpg", "*.npy", "*.bin"]:
            for file in v1_frame_dir.glob(ext):
                dest_file = merge_frame_dir / file.name
                shutil.copy2(file, dest_file)

        # 复制第二个版本的.jpg文件
        for file in v2_frame_dir.glob("*.jpg"):
            dest_file = merge_frame_dir / file.name
            # 如果文件已存在（从v1复制的），添加后缀
            if dest_file.exists():
                dest_file = merge_frame_dir / f"{file.stem}_v2{file.suffix}"
            shutil.copy2(file, dest_file)

        # 保存合并后的JSON
        json3_path = merge_frame_dir / "meta_info.json"
        with open(json3_path, "w", encoding="utf-8") as f:
            json.dump(json3, f, indent=2, ensure_ascii=False)

        return True, filtered_count, has_unmatched

    except Exception as e:
        print(f"  错误: 处理 {merge_frame_dir} 时出错: {e}")
        return False, 0, False


def merge_datasets(
    common_frames: Set[str], version1_path: Path, version2_path: Path, merge_path: Path, dataset_type: str
):
    """
    合并数据集
    """
    print(f"\n开始合并 {len(common_frames)} 个frames...")

    success_count = 0
    fail_count = 0
    total_filtered_objects = 0
    unmatched_files_count = 0  # 统计有多少个文件的json2中存在未匹配的object
    unmatched_files_list = []  # 存储所有有未匹配object的json2文件路径

    for idx, frame_rel_path in enumerate(sorted(common_frames), 1):
        if idx % 100 == 0:
            print(f"进度: {idx}/{len(common_frames)} ({idx*100//len(common_frames)}%)")

        success, filtered_count, has_unmatched = copy_files_for_frame(
            frame_rel_path, version1_path, version2_path, merge_path, dataset_type
        )
        if success:
            success_count += 1
            total_filtered_objects += filtered_count
        else:
            fail_count += 1
            
        if has_unmatched:
                unmatched_files_count += 1
                # 记录json2文件路径
                json2_path = version2_path / frame_rel_path / "meta_info.json"
                unmatched_files_list.append(str(json2_path))

    print(f"\n合并完成!")
    print(f"成功复制: {success_count} 个frames")
    print(f"跳过: {fail_count} 个frames (meta_info.json不存在或ground_info为空)")
    print(f"被过滤的objects总数: {total_filtered_objects}")
    print(f"  (包含'指定的边界框内没有可描述的物体'或'!!!!!!!'的objects)")
    print(f"\n\033[93m警告: 有 {unmatched_files_count} 个文件的json2中存在找不到匹配的object\033[0m")
    
    # 将未匹配的文件列表输出到txt文件
    if unmatched_files_list:
        output_txt = merge_path.parent / f"unmatched_json2_files_{merge_path.name}.txt"
        with open(output_txt, "w", encoding="utf-8") as f:
            for file_path in unmatched_files_list:
                f.write(file_path + "\n")
        print(f"未匹配的json2文件列表已保存到: {output_txt}")
    
    print(f"")


def main():


    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="合并两个版本的数据集")
    parser.add_argument(
        "--version1",
        type=str,
        # required=True,
        default="/vepfs-cnbj70aef516b63d/rongli/code/3eed/data/3eed",
        help="第一个版本的数据集路径",
    )
    parser.add_argument(
        "--version2",
        type=str,
        # required=True,
        default="/vepfs-cnbj70aef516b63d/rongli/code/3eed/data/3eed_labeled_v7",
        # default="/mnt/data3/rongli/3eed_labeled_v8",
        help="第二个版本的数据集路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        # required=True,
        default="/mnt/data3/rongli/3eed_merge",
        help="合并后的输出路径",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["waymo", "drone", "quad"],
        # default="waymo",
        # default="drone",
        default="quad",
        help="数据集类型 (默认: waymo)",
    )

    args = parser.parse_args()

    # 构建完整路径
    version1_path = Path(args.version1) / args.dataset
    version2_path = Path(args.version2) / args.dataset
    merge_path = Path(args.output) / args.dataset

    assert os.path.exists(version1_path), f"第一个版本路径不存在: {version1_path}"
    assert os.path.exists(version2_path), f"第二个版本路径不存在: {version2_path}"
    # if os.path.exists(merge_path):
    #     print(f"警告: 输出路径已存在: {merge_path}")
    #     exit()
    # else:
    #     os.makedirs(merge_path, exist_ok=True)

    print("=" * 80)
    print(f"开始合并{args.dataset}数据集")
    print("=" * 80)
    print(f"第一个版本路径: {version1_path}")
    print(f"第二个版本路径: {version2_path}")
    print(f"输出路径: {merge_path}")

    # 第一步：比较数据集
    print("\n第一步: 检查两个版本的数据一致性...")
    common_frames, only_v1, only_v2 = compare_datasets(version1_path, version2_path)

    # 如果有缺失，询问是否继续
    if only_v1:
        print("\n警告: 第一个版本中有些frame在第二个版本中不存在!")
        print(only_v1)
        # ipdb.set_trace()

    if only_v2:
        print("\n警告: 第二个版本中有些frame在第一个版本中不存在!")
        print(only_v2)
        # ipdb.set_trace()

    if not common_frames:
        print("错误: 没有找到共同的frames!")
        return

    # 第二步：合并数据
    # exit()
    print("\n第二步: 合并数据...")
    merge_datasets(common_frames, version1_path, version2_path, merge_path, args.dataset)

    print("\n完成!")


if __name__ == "__main__":
    main()
