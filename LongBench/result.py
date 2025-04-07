import os, json

files = [f for f in os.listdir('results') if f.endswith('.jsonl')]
output = ["Model\tOverall\tEasy\tHard\tShort\tMedium\tLong\tAvgTokenCount"]
compensated = False

for file in files:
    filename = os.path.join('results', file)
    try:
        pred_data = [json.loads(line) for line in open(filename, encoding='utf-8')]
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        pred_data = []
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    token_count_sum = 0
    for pred in pred_data:
        acc = int(pred['judge'])
        if compensated and pred["pred"] == None:
            acc = 0.25
        if pred["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        if pred['length'] == "short":
            short += 1
            short_acc += acc
        elif pred['length'] == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc
        
        token_count_sum += pred.get('cot_token_count', 0)

    name = '.'.join(file.split('.')[:-1])
    # 避免除零错误，检查分母是否为零
    if len(pred_data) > 0:
        overall = round(100*(easy_acc+hard_acc)/len(pred_data), 1)
    else:
        overall = 0.0
    
    easy_result = round(100*easy_acc/easy, 1) if easy > 0 else 0.0
    hard_result = round(100*hard_acc/hard, 1) if hard > 0 else 0.0
    short_result = round(100*short_acc/short, 1) if short > 0 else 0.0
    medium_result = round(100*medium_acc/medium, 1) if medium > 0 else 0.0
    long_result = round(100*long_acc/long, 1) if long > 0 else 0.0
    avg_token_count = round(token_count_sum/len(pred_data), 1) if len(pred_data) > 0 else 0.0
    
    output.append(f"{name}\t{overall}\t{easy_result}\t{hard_result}\t{short_result}\t{medium_result}\t{long_result}\t{avg_token_count}")

# Sort the output list by model name (excluding the header)
output[1:] = sorted(output[1:])

open('result.txt', 'w', encoding='utf-8').write('\n'.join(output))