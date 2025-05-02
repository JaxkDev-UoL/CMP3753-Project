import json

# Load the ABCD dataset from a local file
def load_abcd_dataset_from_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

actions = []

def format_action(action):
    # If action starts with FAQ, remove it
    if action.startswith('FAQ'):
        return

    # if action ends with 'has been noted.' remove it
    if action.endswith('has been noted.'):
        return

    if action.endswith('have been entered.'):
        return

    if action.startswith('Agent is looking for solutions'):
        return

    if action.endswith('has been recorded.'):
        return
    
    if action.startswith('System Action: search '):
        return

    if action.startswith('Querying the system for '):
        return
    
    if action.startswith('Purchase validation '):
        return

    if action.startswith('Searching the FAQ pages'):
        return
    
    if action.startswith('Identity verification'):
        return
    
    if action in ['A link will be sent.', 'A password has been generated.', 'A promo code has been created.']:
        return
    
    # if action is 'A purchase of * was made.'
    if action.startswith('A purchase of') and action.endswith('was made.'):
        action = '<<purchase_item>>'

    if action.startswith('Account has been updated'):
        action = '<<account_update>>'
    
    if action.startswith('Account has been pulled up'):
        action = '<<account_pulled>>'
    
    if action.startswith('A refund has been made for the amount of '):
        action = '<<refund>>'
    
    if action.endswith('has been notified.'):
        action = '<<send_notification>>'
    
    if action.startswith('Order has been updated with'):
        action = '<<order_update>>'
    

    if action not in actions:
        actions.append(action)
    
    return action

# Process conversation and format for LLaMA
def process_conversation_llama(convo):
    processed_entries = []
    for turn in convo['original']:
        speaker, utterance = turn[0], turn[1]
        # Handle action turns
        if speaker == 'action':
            speaker = 'agent'
            action = format_action(utterance.strip())
            if not action:
                continue
            utterance = action
        # Determine the role
        role = 'user' if speaker == 'customer' else 'assistant'
        processed_entries.append({'role': role, 'content': utterance.strip()})
    
    # Skip leading assistant messages until the first user
    start_index = None
    for idx, entry in enumerate(processed_entries):
        if entry['role'] == 'user':
            start_index = idx
            break
    if start_index is None:  # No user messages found
        return []
    filtered_entries = processed_entries[start_index:]
    
    # Merge consecutive entries with the same role
    merged = []
    current_role = None
    current_content = []
    for entry in filtered_entries:
        if entry['role'] == current_role:
            current_content.append(entry['content'])
        else:
            if current_role is not None:
                merged.append({'role': current_role, 'content': '\n'.join(current_content)})
            current_role = entry['role']
            current_content = [entry['content']]
    if current_role is not None:
        merged.append({'role': current_role, 'content': '\n'.join(current_content)})
    
    # Ensure the conversation ends with an assistant
    if merged and merged[-1]['role'] == 'user':
        merged.append({'role': 'assistant', 'content': ''})
    
    # Pair user and assistant turns
    paired_data = []
    i = 0
    while i < len(merged):
        if merged[i]['role'] != 'user':
            i += 1
            continue
        user_content = merged[i]['content']
        assistant_content = ''
        if i + 1 < len(merged) and merged[i+1]['role'] == 'assistant':
            assistant_content = merged[i+1]['content']
            i += 2
        else:
            i += 1
        paired_data.append({
            'user': user_content,
            'assistant': assistant_content
        })
    
    return paired_data

def process_conversation_olmo(convo):
    #todo
    return convo

# Process the entire dataset
data = load_abcd_dataset_from_file('datasets/abcd/abcd_v1.1.json')
data = data['train'] + data['dev'] + data['test']

# processed_conversations = []
# for type in processed_conversations.keys():
#     for convo in data[type]:
#         processed_conversations.append((process_conversation_llama(convo['original']),))
#     f = open('datasets/abcd/abcd_v1.1_processed.json', 'w')
#     json.dump(processed_conversations[type], f, indent=4)
#     f.close()

# json.dump(actions, open('datasets/abcd/abcd_v1.1_actions.json', 'w'), indent=4)

# print('Done processing ABCD dataset, saved to abcd_v1.1_processed_train/dev/test.json - (Conversations: {})'.format(len(processed_conversations['train']) + len(processed_conversations['dev']) + len(processed_conversations['test'])))

with open('datasets/abcd/abcd_v1.1_processed.jsonl', 'w') as out_f, open('datasets/abcd/abcd_v1.1_tokens.json', 'w') as token_f:
    convos = [process_conversation_llama(conv) for conv in data]

    # Write all formatted conversations to file
    # Each line is a JSON object representing a conversation
    for conv in convos:
        out_f.write(json.dumps(conv) + "\n")

    # Write special tokens to token file
    json.dump(sorted(actions), token_f)