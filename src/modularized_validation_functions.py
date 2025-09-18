# Notes
'''
Author: Gyumin Lee
Version: 2.0
Description (primary changes): Code refactoring
'''

import torch
import numpy as np
import copy
import pandas as pd
from utils import to_device

# These variables are expected to be defined in the notebook environment
# tech_dataset, used_test_index_TC, used_test_data_TC, used_rawdata, total_data, n_TC

def prepare_input_data(tech_dataset, used_test_index_TC, idx, model):
    # Prepares input data and loads it to the model device
    # Encode class and claim data
    input_class = torch.tensor(tech_dataset.tokenizers["class_enc"].encode(tech_dataset.X_class[used_test_index_TC][idx])).unsqueeze(0)
    input_claim = tech_dataset.tokenize(tech_dataset.tokenizers["claim_enc"], tech_dataset.X_claim[used_test_index_TC][idx])
    input_claim = {k: v.unsqueeze(0) for k, v in input_claim.items()}
    
    # Create batch input and send to model device
    batch_input = {"class": torch.tensor(input_class), "claim": input_claim}
    input_inf = to_device(batch_input, model.device)
    
    return input_inf, input_class

def encode_and_predict(model, input_inf):
    # Encodes input data and performs initial prediction
    enc_outputs, z, mu, logvar = model.encode(input_inf)
    pred_outputs = model.predict(z)
    
    return enc_outputs, z, mu, logvar, pred_outputs

def analyze_forward_references(used_test_data_TC, used_rawdata, total_data, idx, n_TC):
    # Analyzes forward references to retrieve IPC codes and citation counts of referenced patents
    # Check if forward citations exist
    if used_test_data_TC.iloc[idx]["TC5"] <= 0:
        return False, None, None
    
    # Get forward reference information
    forward_refs = used_rawdata.loc[used_test_data_TC.iloc[idx]["patent_number"]]["forward_refs"].split(";")
    ref_info = total_data.loc[[ref for ref in forward_refs if ref in total_data.index]]
    
    if len(ref_info) == 0:
        return False, None, None
        
    # Get IPC codes and citation counts of referenced patents
    ref_ipcs = ref_info["patent_classes"].apply(lambda x: set(x))
    ref_FCs = ref_info["TC" + str(n_TC)]
    
    return True, ref_ipcs, ref_FCs

def decode_original_text(tech_dataset, input_class):
    # Decodes the original IPC code from input class tensor
    tokenizer = tech_dataset.tokenizers["class_dec"]
    org_text = tokenizer.decode_batch(input_class.cpu().detach().numpy())[0]
    org_text = org_text[org_text.index(tokenizer.sos_token)+1:org_text.index(tokenizer.eos_token)]
    
    return org_text

def check_same_ipcs(org_text, ref_ipcs):
    # Checks if the original IPC codes are the same as those in referenced patents
    return set(org_text) == set(np.concatenate(ref_ipcs.apply(lambda x: list(x)).values))

def optimize_latent_space(model, z, enc_outputs, tech_dataset, L1_threshold, n_iter, step_size):
    # Optimizes latent space through gradient descent to generate IPC codes with citation count above L1_threshold
    tokenizer = tech_dataset.tokenizers["class_dec"]
    optimized = False
    gen_text = None
    FC_estimated = 0
    
    # Iterative optimization using gradient descent
    for i in range(n_iter):
        pred_outputs = model.predict(z)
        z.retain_grad()
        FC_estimated = np.round(np.exp(pred_outputs[0,1].item()), 4)  # estimated forward citations
        
        # Calculate L1 error and backpropagate
        L1_error = (1 - torch.exp(pred_outputs[0,1]))
        L1_error.backward(retain_graph=True)
        
        # Update latent vector in gradient direction
        grad_for_update = step_size * z.grad
        z_ = z - grad_for_update
        
        z.grad.zero_()
        
        # Generate IPC code with new latent vector
        dec_outputs = model.decode(z_, enc_outputs, dec_inputs=None)
        dec_outputs = dec_outputs.argmax(-1)
        
        # Decode generated IPC code
        gen_text = tokenizer.decode_batch(dec_outputs.cpu().detach().numpy())[0]
        if tokenizer.eos_token in gen_text:
            gen_text = gen_text[gen_text.index(tokenizer.sos_token)+1:gen_text.index(tokenizer.eos_token)]
        else:
            gen_text = gen_text[gen_text.index(tokenizer.sos_token)+1:]
            
        # Refine generated IPC code
        if gen_text != []:
            gen_text = [gen_text[0]] + list(np.array(gen_text[1:])[np.unique(gen_text[1:], return_index=True)[1]])
            gen_text = set(gen_text)
        else:
            continue
        
        # End optimization if estimated citation count is sufficient
        if FC_estimated >= L1_threshold:
            optimized = True
            break
            
        z = z_
        
    return optimized, gen_text, FC_estimated, z

def breakdown(ipcs):
    # Breaks down IPC codes into different levels (section, class, subclass, full)
    return ([ipc[0] for ipc in ipcs], [ipc[:3] for ipc in ipcs], [ipc[:4] for ipc in ipcs], ipcs)

def validate_generated_ipc(gen_text, ref_ipcs, ref_FCs):
    # Validates generated IPC codes by comparing with IPC codes of referenced patents
    inclusions = [None, None, None, None]
    higher_impacts = [None, None, None, None]
    similar_refs = [None, None, None, None]
    unsimilar_refs = [None, None, None, None]
    
    # Break down IPC codes
    gen_text_breakdown = breakdown(gen_text)
    ref_ipcs_breakdown = (
        ref_ipcs.apply(lambda x: breakdown(x)[0]),
        ref_ipcs.apply(lambda x: breakdown(x)[1]),
        ref_ipcs.apply(lambda x: breakdown(x)[2]),
        ref_ipcs
    )
    
    # Validate at each level
    for i in range(4):
        if inclusions[i] is not None:
            continue
            
        temp_gen_text = gen_text_breakdown[i]
        temp_ref_ipcs = ref_ipcs_breakdown[i]
        
        # Find references matching generated IPC
        hit_index = temp_ref_ipcs.apply(lambda x: 1 if set(x) == set(temp_gen_text) else 0) == 1
        similar_refs[i] = temp_ref_ipcs[hit_index].index
        unsimilar_refs[i] = temp_ref_ipcs[~hit_index].index
        
        # No matching references
        if len(similar_refs[i]) == 0:
            inclusions[i] = 0
            higher_impacts[i] = None
        # All references match
        elif len(unsimilar_refs[i]) == 0:
            inclusions[i] = 1
            similar_mean_FC = np.mean(ref_FCs.loc[similar_refs[i]])
            higher_impacts[i] = 1 if similar_mean_FC > 0 else 0
        # Some references match
        else:
            inclusions[i] = 1
            similar_mean_FC = np.mean(ref_FCs.loc[similar_refs[i]])
            unsimilar_mean_FC = np.mean(ref_FCs.loc[unsimilar_refs[i]])
            
            if similar_mean_FC >= unsimilar_mean_FC:
                higher_impacts[i] = 1 if similar_mean_FC > 0 else None
            else:
                higher_impacts[i] = 0
    
    return inclusions, higher_impacts, similar_refs, unsimilar_refs

def validate_reliability(model=None, idx=None, L1_threshold=0.5, n_iter=30, step_size=40):
    # Validates the reliability of generated IPC codes
    # Initialize counters
    cnt_nonexist = 0
    cnt_noFC = 0
    cnt_diverge = 0
    cnt_same_ipcs = 0
    cnt_diff_ipcs = 0
    
    # 1. Prepare input data
    input_inf, input_class = prepare_input_data(tech_dataset, used_test_index_TC, idx, model)
    
    # 2. Encode and perform initial prediction
    enc_outputs, z, mu, logvar, pred_outputs = encode_and_predict(model, input_inf)
    
    # 3. Analyze forward references
    ref_exists, ref_ipcs, ref_FCs = analyze_forward_references(used_test_data_TC, used_rawdata, total_data, idx, n_TC)
    
    if not ref_exists:
        if used_test_data_TC.iloc[idx]["TC5"] <= 0:
            cnt_noFC += 1
        else:
            cnt_nonexist += 1
        return (cnt_nonexist, cnt_noFC, cnt_diverge, cnt_same_ipcs, cnt_diff_ipcs), None
    
    # 4. Decode original IPC code
    org_text = decode_original_text(tech_dataset, input_class)
    
    # 5. Check if original IPC matches referenced IPCs
    if check_same_ipcs(org_text, ref_ipcs):
        cnt_same_ipcs += 1
    
    # 6. Generate IPC code through latent space optimization
    optimized, gen_text, FC_estimated, z = optimize_latent_space(model, z, enc_outputs, tech_dataset, L1_threshold, n_iter, step_size)
    
    if not optimized:
        cnt_diverge += 1
        return (cnt_nonexist, cnt_noFC, cnt_diverge, cnt_same_ipcs, cnt_diff_ipcs), None
    
    # 7. Validate generated IPC code
    inclusions, higher_impacts, similar_refs, unsimilar_refs = validate_generated_ipc(gen_text, ref_ipcs, ref_FCs)
    
    # 8. Collect final results
    cnt_diff_ipcs += 1
    results = {
        "index": idx,
        "patent_id": used_test_data_TC.iloc[idx]["patent_number"],
        "org_text": org_text,
        "gen_text": gen_text,
        "ref_ipcs": ref_ipcs,
        "ref_FCs": ref_FCs,
        "inclusions": inclusions,
        "higher_impacts": higher_impacts, 
        "FC_estimated": FC_estimated,
        "similar_refs": similar_refs,
        "unsimilar_refs": unsimilar_refs
    }
    
    return (cnt_nonexist, cnt_noFC, cnt_diverge, cnt_same_ipcs, cnt_diff_ipcs), results
