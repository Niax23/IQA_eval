#!/bin/bash

# 设置输入图像和参考图像路径
setting="FullIR"
dataset="RealSR"
method="Depict27k_realprompt"
test_image_path="/home/jiyifei/MSR_results/${setting}/${dataset}/${method}"
ref_image_path="/home/jiyifei/MSR_results/${setting}/${dataset}/gt"

# 输出结果的路径
output_dir="./evaluation_results/${setting}/${dataset}/${method}/"
mkdir -p "$output_dir"

# 指标列表
#metrics=("dists")

python psnr_ssim_eval.py --img_dir $test_image_path --gt_dir $ref_image_path --output_dir $output_dir
metrics=("lpips" "dists" "fid" "niqe" "maniqa" "musiq" "clipiqa")
export CUDA_VISIBLE_DEVICES=2,
# 遍历每个指标并运行对应的命令
for metric in "${metrics[@]}"
do
    echo "Evaluating $metric..."

    # 运行 pyiqa 评估命令
    output_file="$output_dir/${metric}_result.txt"

    # 根据不同指标调用 pyia 命令
    case "$metric" in
        "lpips")
            pyiqa lpips -t "$test_image_path" -r "$ref_image_path" --device cuda --verbose > "$output_file"
            ;;
        "dists")
            pyiqa dists -t "$test_image_path" -r "$ref_image_path" --device cuda --verbose > "$output_file"
            ;;
        "fid")
            pyiqa fid -t "$test_image_path" -r "$ref_image_path" --device cuda --verbose > "$output_file"
            ;;
        "niqe")
            pyiqa niqe -t "$test_image_path" -r "$ref_image_path" --device cuda --verbose > "$output_file"
            ;;
        "maniqa")
            pyiqa maniqa-pipal -t "$test_image_path"  --device cuda --verbose > "$output_file"
            ;;
        "musiq")
            pyiqa musiq -t "$test_image_path"  --device cuda --verbose > "$output_file"
            ;;
        "clipiqa")
            pyiqa clipiqa -t "$test_image_path"  --device cuda --verbose > "$output_file"
            ;;
        *)
            echo "Unknown metric: $metric"
            ;;
    esac

    # 输出已完成的指标
    echo "$metric evaluation done. Results saved to $output_file"
done




echo "All evaluations completed. Results saved in $output_dir"
