import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    return parser.parse_args()


def merge_multiple_jsonl(input_folder, save_path):
    merged_data = []

    # 입력 폴더에서 모든 JSONL 파일 읽기
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(input_folder, file_name)
            data = read_jsonl(file_path)
            merged_data.extend(data)

    # 합쳐진 데이터를 새로운 파일에 쓰기
    write_jsonl(save_path, merged_data)


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 각 라인의 JSON 문자열을 파싱하여 리스트에 추가
            data.append(json.loads(line))
    return data


def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            # 각 JSON 객체를 문자열로 변환하여 파일에 쓰기
            file.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    args = parse_args()
    merge_multiple_jsonl(args.input_folder, args.save_path)


if __name__ == "__main__":
    main()
