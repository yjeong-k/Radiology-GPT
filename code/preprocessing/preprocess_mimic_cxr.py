import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type = str, required = True)
    parser.add_argument("--save_path", type = str, required=True)
    return parser.parse_args()

def preprocess_note(notes):
    res = {}
    subjects = [
        "EXAMINATION", "HISTORY", "INDICATION", "TECHNIQUE", 
        "COMPARISON", "FINDINGS", "IMPRESSION"
        ]
    for subject in subjects:
        res[subject] = ""
    
    notes = notes.splitlines()
    prevkey = ""
    for line in notes:
        idx = line.find(":")
        if idx != -1:
            # contains ":"
            key = line[:idx].strip()
            prevkey = key
            value = line[idx+1:].strip()
            res[key] = value
        else:
            if line.isupper() or line.strip() == "" or prevkey=="":
                continue
            else:
                prev_value = res[prevkey]
                res[prevkey] = prev_value + line
    return res

def main():
    args = parse_args()
    df_path = pd.read_csv(args.input_path)
    df_result = pd.DataFrame(columns=['id', 'note'])
    i = 0
    for path in df_path["path"]:
        with open(path) as f:
            note_lines = f.read()
            res = preprocess_note(note_lines)
        id = df_path['study_id'].iloc[i]
        df_temp  = pd.DataFrame.from_dict([res]).reset_index()
        df_temp['id'] = id
        df_temp = df_temp[
            [
                "id","EXAMINATION", "HISTORY", "INDICATION", "TECHNIQUE", 
                "COMPARISON", "FINDINGS", "IMPRESSION"
                ]
            ]
        df_result.loc[i] = [id, note_lines]


        i +=1
        if i% 1000 == 0:
            print(f"{i} files has preprocessed")
            
    df_result.to_csv(args.save_path)

if __name__ == "__main__":
    main()