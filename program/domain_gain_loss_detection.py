from collections import defaultdict
import json
from sklearn.cluster import DBSCAN
import numpy as np
import os


# *************************************  common functions  ************************************************
def pf_dict_load(json_file):
    with open(json_file, 'r') as fin:
        pf_dict = json.load(fin)
    return pf_dict


def convert_str_to_list(value_list):
    str_list = str()
    for value in value_list:
        str_list += str(value)
    return str_list


def convert_list_to_str(position_list):
    result = str()
    for pos in position_list:
        result += str(pos)
    return result


def convert_pos_to_dis(node, short_distance_dict):
    pos = str()
    dis_list = list()
    for char in node:
        pos += char
        dis = (2*int(char)-1)*short_distance_dict[pos]
        dis_list.append(dis)
    return dis_list


def tag_decision(tag_left, tag_right):
    if tag_left == tag_right:
        tag = tag_left
    elif tag_left == '?':
        tag = tag_right
    elif tag_right == '?':
        tag = tag_left
    else:
        tag = '?'
    return tag


# *************************************  Phylogenetic tree  *********************************************
def calculate_long_distance(short_distance_dict, long_distance_dict):
    nodes = list(short_distance_dict.keys())
    nodes.sort(key=lambda i: len(i), reverse=False)
    for node in nodes:
        if len(node) == 1:
            long_distance_dict[node] = short_distance_dict[node]
        else:
            long_distance_dict[node] = short_distance_dict[node] + long_distance_dict[node[:-1]]


def nj_tree_post_process(ref, name_position, long_distance_dict, spe_tax, short_distance_dict):
    short_distance_dict['0'] = 0
    long_distance_dict['0'] = 0
    with open(ref, "r") as phy_file:
        position_list = [0]
        name = str()
        distance = str()

        # state: 0 => node; 1 => name; 2 => distance; 3 => mid_state
        state = 0
        content = phy_file.readline()
        for char in content:
            # FSM
            if state == 0 and char == '(':
                position_list.append(0)
            elif state == 0 and (char.isalpha() or char.isdigit() or char == '_' or char == '\'' or char == '.' or char == '-'):
                state = 1
                name = name + char
            elif state == 1 and (char.isalpha() or char.isdigit() or char == '_' or char == '\'' or char == '.' or char == '-'):
                name = name + char
            elif state == 1 and char == ':':
                state = 2
                # spe_id = name.split('_')[-1]
                if '\'' in name:
                    name = name.split('\'')[1]
                spe_id = name
                tax_name = '_'.join(name.split('_')[:-1])
                spe_tax[spe_id] = tax_name
                name_position[spe_id] = convert_list_to_str(position_list.copy())
                name = ''
            elif state == 2 and (char.isdigit() or char == '.' or char == '-' or char == 'E'):
                distance += char
                continue
            elif state == 2 and char == ')':
                pos_value = convert_list_to_str(position_list)
                short_distance_dict[pos_value] = float(distance)
                state = 3
                position_list.pop()
                distance = ''
            elif state == 2 and char == ',':
                pos_value = convert_list_to_str(position_list)
                short_distance_dict[pos_value] = float(distance)
                position_list[-1] += 1
                state = 0
                distance = ''
            elif state == 3 and char == ':':
                state = 2
            elif state == 3 and char == ';':
                print("End of process for newick file!")
                break
            else:
                print("Illegal state!")
                print(state)
                print(char)
                return
        calculate_long_distance(short_distance_dict, long_distance_dict)
    return


# ****************************************  MP  ********************************************
def iter_search(mid_node_tag, all_node, top_node, node, tag, leaf_ref, all_node_dict, leaf_node_dict):
    all_node.remove(node)
    all_node_dict.setdefault(top_node, list()).append(node)
    if node in leaf_ref:
        leaf_node_dict.setdefault(top_node, list()).append(node)
    child_left = node + '0'
    child_right = node + '1'
    if child_left in mid_node_tag.keys() and mid_node_tag[child_left] == tag:
        iter_search(mid_node_tag, all_node, top_node, child_left, tag, leaf_ref, all_node_dict, leaf_node_dict)
    if child_right in mid_node_tag.keys() and mid_node_tag[child_right] == tag:
        iter_search(mid_node_tag, all_node, top_node, child_right, tag, leaf_ref, all_node_dict, leaf_node_dict)


def find_distance(re_node, ref_node, long_distance_dict):
    a = long_distance_dict[re_node]
    b = long_distance_dict[ref_node]
    c = 0
    for i in range(min(len(ref_node), len(re_node))):
        if re_node[i] != ref_node[i]:
            c = long_distance_dict[re_node[:i]]
            break
    return a+b-2*c


def top_node_tax_included(leaf_dict, pos_tax):
    top_node_leaf_tax = dict()
    for leaf in leaf_dict:
        leaf_tax = pos_tax[leaf]
        if leaf_tax not in top_node_leaf_tax.keys():
            top_node_leaf_tax[leaf_tax] = 1
        else:
            top_node_leaf_tax[leaf_tax] += 1
    return top_node_leaf_tax


def mp_based_tagging_algorithm(name_position, pf_dict, spe_tax, short_distance_dict, pf_id, pos_tax, gain_pf_str_list, loss_pf_str_list, never_pf_str_list, if_transfer_network, transfer_network_output, domain, itol_branch_symbol):
    mid_node_tag = dict()
    # tag
    count_0 = 0
    count_1 = 0
    for leaf_node in spe_tax.keys():
        if leaf_node in pf_dict[pf_id]:
            mid_node_tag[name_position[leaf_node]] = '1'
            count_1 += 1
        else:
            mid_node_tag[name_position[leaf_node]] = '0'
            count_0 += 1

    # from bottom to top
    all_nodes = sorted(short_distance_dict.keys(), key=lambda i: len(i), reverse=True)
    for node in all_nodes:
        if node not in mid_node_tag.keys():
            node_left = node + '0'
            node_right = node + '1'
            tag_left = mid_node_tag[node_left]
            tag_right = mid_node_tag[node_right]
            tag = tag_decision(tag_left, tag_right)
            mid_node_tag[node] = tag
    # root node
    if mid_node_tag['0'] == '?':
        if count_1 > count_0:
            mid_node_tag['0'] = '1'
        else:
            mid_node_tag['0'] = '0'
    # from top to bottom: '?'
    all_nodes = sorted(short_distance_dict.keys(), key=lambda i: len(i))
    for node in all_nodes:
        if mid_node_tag[node] == '?':
            tag = mid_node_tag[node[:-1]]
            assert(tag != '?')
            mid_node_tag[node] = tag

    # clustering
    # include: is_dict[top_node: '100'] = [node_list]
    # exclude: not_dict[top_node: '110'] = [node_list]
    is_dict = dict()
    is_leaf_dict = dict()
    not_dict = dict()
    not_leaf_dict = dict()
    while len(all_nodes) > 0:
        node = all_nodes[0]
        if mid_node_tag[node] == '1':
            iter_search(mid_node_tag, all_nodes, node, node, '1', name_position.values(), is_dict, is_leaf_dict)
        else:
            iter_search(mid_node_tag, all_nodes, node, node, '0', name_position.values(), not_dict, not_leaf_dict)

    if len(is_dict.keys()) > 0:
        # gain group
        is_top_list = list(is_dict.keys())
        gain_pf_str_list[pf_id] = is_top_list

        # loss and never exist groups
        loss_group = list()
        not_group = list()
        not_top_list = list(not_dict.keys())
        for top_node in not_top_list:
            not_group.append(top_node)
            for is_top_node in is_top_list:
                length = len(is_top_node)
                if is_top_node == top_node[:length]:
                    loss_group.append(top_node)
                    not_group.remove(top_node)
                    break
        # if len(loss_group) > 0:
        loss_pf_str_list[pf_id] = loss_group
        # if len(not_group) > 0:
        never_pf_str_list[pf_id] = not_group

        if if_transfer_network and pf_id == domain:
            print('Generating transfer network for domain {}'.format(domain))
            for node in loss_group:
                top_node_tax_included(not_leaf_dict[node], pos_tax)
            # print('Never exist node group: ')
            for node in not_group:
                top_node_tax_included(not_leaf_dict[node], pos_tax)

            all_list = is_top_list + loss_group + not_group
            all_list_sorted = sorted(all_list, key=lambda i: len(i), reverse=True)
            length = len(all_list_sorted)
            with open(transfer_network_output, 'w') as fin:
                with open(itol_branch_symbol, 'r') as fout:
                    for line in fout:
                        fin.write(line)
                    fin.write('\n')
                for i in range(length):
                    node_po = all_list_sorted[i]
                    for j in range(i+1, length):
                        node_ref = all_list_sorted[j]
                        if node_po[:len(node_ref)] == node_ref and node_po != node_ref:
                            if node_po in is_top_list:
                                color = '#00ff00'
                            elif node_po in loss_group:
                                color = '#ff0000'
                            # leaf node
                            if node_po in pos_tax.keys():
                                name = pos_tax[node_po]
                            # middle node
                            else:
                                key_leaf = node_po + '0'
                                while key_leaf not in pos_tax.keys():
                                    key_leaf += '0'
                                key_right = node_po + '1'
                                while key_right not in pos_tax.keys():
                                    key_right += '1'
                                name = pos_tax[key_leaf] + '|' + pos_tax[key_right]
                            fin.write(name + ',2,4,' + color + ',1,0\n')
                            break
    else:
        print(pf_id)


# **************************************  main process  ***********************************************
def relation_collection(str_pf_list_t, domain_file):
    # record[domain_combinations] = [per, number_c, number_1, ... number_n]
    record = dict()
    domain_combinations_list = list()
    domain_combinations_switch = list()

    file_tmp = domain_file.replace('domain_combination', '2_domain')
    file_tmp = file_tmp.replace('tsv', 'tmp')
    print('==> 2-domains combinations')
    # start from the combinations including two domains
    domain_list = list()
    str_pf_list = str_pf_list_t.copy()
    if '0' in str_pf_list.keys():
        str_pf_list.pop('0')
    for key in str_pf_list.keys():
        domain_list += str_pf_list[key]
    if not os.path.exists(file_tmp):
        domain_set = set(domain_list)
        length = len(domain_set)
        cc = 0
        for domain in domain_set:
            cc += 1
            if cc % 100 == 0:
                print('Processing ' + str(cc) + ' of ' + str(length))
            domain_list_tmp = list()
            for tmp in str_pf_list.values():
                if domain in tmp:
                    domain_list_tmp += tmp
            domain_set_temp = set(domain_list_tmp)

            for ref_domain in domain_set_temp:
                if ref_domain != domain:
                    number_1 = domain_list_tmp.count(domain)
                    number_c = domain_list_tmp.count(ref_domain)
                    number_2 = domain_list.count(ref_domain)
                    per = round(number_c/(number_1+number_2-number_c), 3)
                    if per >= 1 and number_c >= 5:
                        tmp_list = [domain, ref_domain]
                        if '-'.join([ref_domain, domain]) not in record.keys():
                            record['-'.join(tmp_list)] = [per, number_c, number_1, number_2]
                            domain_combinations_list.append(tmp_list)
        json_tmp = json.dumps(record)
        with open(file_tmp, 'w') as fout:
            fout.write(json_tmp)
    else:
        with open(file_tmp, 'r') as json_input:
            record = json.load(json_input)
        for key in record.keys():
            domain_combinations_list.append(key.split('-'))

    print('==> multi-domains combinations')
    # find the results by iteration
    remove_dup = list()
    count_a = 3
    while len(domain_combinations_list) != 0:
        print("{} domains combination".format(str(count_a)))
        count_a += 1
        length = len(domain_combinations_list)
        cc = 0
        for domain_combinations in domain_combinations_list:
            cc += 1
            if cc % 100 == 0:
                print('Processing ' + str(cc) + ' of ' + str(length))
            for domain_ref_list in str_pf_list.values():
                if set(domain_combinations).issubset(set(domain_ref_list)):
                    for a in set(domain_ref_list).difference(set(domain_combinations)):
                        record_value = record['-'.join(domain_combinations)].copy()
                        number_a = domain_list.count(a)
                        a_plus = domain_combinations.copy()
                        a_plus.append(a)
                        count = 0
                        for ref_a in str_pf_list.values():
                            if set(a_plus).issubset(set(ref_a)):
                                count += 1
                        per = round(count/(sum(record_value[2:]) + number_a - (len(a_plus)-1)*count), 3)
                        if per >= 1 and count >= 5 and set(a_plus) not in remove_dup:
                            remove_dup.append(set(a_plus))
                            record_value[0] = per
                            record_value.append(number_a)
                            record['-'.join(a_plus)] = record_value.copy()
                            domain_combinations_switch.append(a_plus)
        domain_combinations_list = domain_combinations_switch.copy()
        domain_combinations_switch.clear()

    # record in the file
    ref_set_list = list()
    with open(domain_file, 'w') as fout:
        fout.write('domain_combinations\tpercentage\tnumber_c\t...\n')
        keys_list = list(record.keys())
        keys_list.sort(key=lambda i: len(i), reverse=True)
        for key in keys_list:
            jump = False
            set_tmp = set(key.split('-'))
            for ref_set in ref_set_list:
                if set_tmp.issubset(ref_set):
                    jump = True
                    break
            if not jump:
                ref_set_list.append(set_tmp)
                fout.write(key)
                fout.write('\t')
                for value in record[key]:
                    fout.write(str(value))
                    fout.write('\t')
                fout.write('\n')


def gain_and_loss_collection(gain_file, loss_file, output):
    gain_dict = dict()
    loss_dict = dict()
    gain_loss_dict = dict()
    with open(gain_file, 'r') as fin:
        for line in fin.readlines():
            if 'domain' not in line:
                key = line.split('\t')[0]
                value = line.split('\t')[1:]
                gain_dict[key] = value
    with open(loss_file, 'r') as fin:
        for line in fin.readlines():
            if 'domain' not in line:
                key = line.split('\t')[0]
                value = line.split('\t')[1:]
                loss_dict[key] = value

    for key in gain_dict.keys():
        for key_2 in loss_dict.keys():
            if set(key.split('-')).issubset(set(key_2.split('-'))):
                gain_loss_dict[key] = gain_dict[key].copy()
    for key in loss_dict.keys():
        for key_2 in gain_dict.keys():
            if set(key.split('-')).issubset(set(key_2.split('-'))):
                gain_loss_dict[key] = loss_dict[key].copy()

    with open(output, 'w') as fout:
        for key in gain_loss_dict.keys():
            fout.write(key)
            fout.write('\t')
            for value in gain_loss_dict[key]:
                fout.write(str(value).replace('\n', ''))
                fout.write('\t')
            fout.write('\n')


def clustering_and_detection(method, name_position, pf_dict, spe_tax, short_distance_dict, pos_tax, domain, json_gain, json_loss, json_never, if_transfer_network, if_itol_sankey, if_number, if_overlaps, transfer_network_output, itol_output, sankey_output, number_output, domain_combination_output, itol_node, itol_branch, domain_limitation_value, itol_branch_symbol):
    # MP: tag and cluster
    if method == 'MP':
        # 0: Prepare middle results for further analysis in terms of single and multiple domains ############
        gain_pf_str_list = dict()
        loss_pf_str_list = dict()
        never_pf_str_list = dict()
        if not (os.path.exists(json_gain) and os.path.exists(json_loss)):
            # record in the json file
            num_domain = len(pf_dict.keys())
            count = 0
            for pf_id in pf_dict.keys():
                if count % 500 == 0:
                    print("Processing domain recombination (MP) " + str(count) + "/{} ...".format(num_domain))
                count += 1
                mp_based_tagging_algorithm(name_position, pf_dict, spe_tax, short_distance_dict, pf_id, pos_tax, gain_pf_str_list, loss_pf_str_list, never_pf_str_list, if_transfer_network, transfer_network_output, domain, itol_branch_symbol)
            json_str = json.dumps(gain_pf_str_list)
            with open(json_gain, 'w') as json_input:
                json_input.write(json_str)
            json_str = json.dumps(loss_pf_str_list)
            with open(json_loss, 'w') as json_input:
                json_input.write(json_str)
            json_str = json.dumps(never_pf_str_list)
            with open(json_never, 'w') as json_input:
                json_input.write(json_str)

        else:
            # read from the json file
            with open(json_gain, 'r') as json_input:
                gain_pf_str_list = json.load(json_input)
            with open(json_loss, 'r') as json_input:
                loss_pf_str_list = json.load(json_input)
            with open(json_never, 'r') as json_input:
                never_pf_str_list = json.load(json_input)
            if if_transfer_network:
                mp_based_tagging_algorithm(name_position, pf_dict, spe_tax, short_distance_dict, domain, pos_tax, gain_pf_str_list, loss_pf_str_list, never_pf_str_list, if_transfer_network, transfer_network_output, domain, itol_branch_symbol)

        # 1 Single domain #####################################
        if if_itol_sankey:
            print('Generating iToL render file for domain {}'.format(domain))
            # 1.1 Create iTOL render file
            # evo_dict[leaf_node] = ["gain", "loss", "gain", "loss", ...]
            evo_dict = dict()
            phy_type_dict = dict()
            gain_list = gain_pf_str_list[domain]
            loss_list = loss_pf_str_list[domain]
            # never_list = never_pf_str_list[domain]
            with open(itol_output, 'w') as fin:
                with open(itol_node, 'r') as fout:
                    for line in fout:
                        fin.write(line)
                fin.write('\n')
                for leaf_name in name_position.keys():
                    leaf_node = name_position[leaf_name]
                    pos_temp = ''
                    for a in leaf_node:
                        evo_dict.setdefault(leaf_node, list())
                        pos_temp += a
                        if pos_temp in gain_list:
                            evo_dict.setdefault(leaf_node, list()).append("gain")
                        elif pos_temp in loss_list:
                            evo_dict.setdefault(leaf_node, list()).append("loss")
                    if len(evo_dict[leaf_node]) == 0:
                        evo_dict.setdefault(leaf_node, list()).append("none")
                    key = spe_tax[pos_tax[leaf_node]] + '->' + evo_dict[leaf_node][-1]
                    if key not in phy_type_dict.keys():
                        phy_type_dict[key] = 1
                    else:
                        phy_type_dict[key] += 1
                    length = len(evo_dict[leaf_node])
                    gap = 10
                    fin.write(leaf_name + ',' + str(gap * length))
                    for i in range(length):
                        type_evo = evo_dict[leaf_node][i]
                        low_num = i*gap
                        up_num = (i+1)*gap
                        if type_evo == "gain":
                            t_color = "#00ff00"
                            t_type = "GAIN"
                        elif type_evo == "loss":
                            t_color = "#ff0000"
                            t_type = "LOSS"
                        else:
                            t_color = "#a9a9a9"
                            t_type = "NONE"
                        fin.write(",RE|" + str(low_num) + "|" + str(up_num) + "|" + t_color + "|" + t_type)
                    fin.write("\n")
            print('Generating Sankey diagram for domain {}'.format(domain))
            # 1.2 Sankey: Showing the distribution of certain domain
            with open(sankey_output, 'w') as fin:
                fin.write("Phyla\tEvents\tNumber\n")
                for i in sorted(phy_type_dict):
                    str_tmp = i.replace("->", "\t") + '\t' + str(phy_type_dict[i]) + '\n'
                    fin.write(str_tmp)

        # 2 multiple domains #####################################
        if if_number or if_overlaps:
            # 2.1 preprocess： **_str_pf_list[node_position] = [pf1, pf2, ...] ###############
            # gain
            gain_str_pf_list = dict()
            for key in gain_pf_str_list.keys():
                for pos in gain_pf_str_list[key]:
                    gain_str_pf_list.setdefault(pos, list()).append(key)
            # loss
            loss_str_pf_list = dict()
            for key in loss_pf_str_list.keys():
                for pos in loss_pf_str_list[key]:
                    loss_str_pf_list.setdefault(pos, list()).append(key)

            if if_number:
                print('Collecting number of each event')
                # 2.2 Number of domain in gain & loss events of each differentiation position
                for str_pf_list in [gain_str_pf_list.copy(), loss_str_pf_list.copy()]:
                    if str_pf_list == gain_str_pf_list:
                        type_name = 'gain'
                    else:
                        type_name = 'loss'
                    type_num_num = dict()
                    for key in str_pf_list.keys():
                        num = len(str_pf_list[key])
                        str_pf_list[key] = num
                        if num not in type_num_num.keys():
                            type_num_num[num] = 1
                        else:
                            type_num_num[num] += 1
                    key_sorted = sorted(type_num_num.keys())
                    with open(number_output.format(type_name), 'w') as fin:
                        fin.write('number_of_domain\tfrequency\n')
                        for key in key_sorted:
                            fin.write(str(key) + '\t' + str(type_num_num[key]) + '\n')

                # Frequency of domain number in gain & loss events of each differentiation position
                gain_dict = dict()
                loss_dict = dict()
                with open(number_output.format('gain'), 'r') as fin:
                    for line in fin.readlines():
                        if 'number' not in line:
                            gain_dict[line.split('\t')[0]] = line.split('\t')[1].split('\n')[0]
                with open(number_output.format('loss'), 'r') as fin:
                    for line in fin.readlines():
                        if 'number' not in line:
                            loss_dict[line.split('\t')[0]] = line.split('\t')[1].split('\n')[0]
                with open(number_output.format('gain_loss'), 'w') as fout:
                    fout.write('number_of_domain\tfrequency_gain\tfrequency_loss\n')
                    key_list = list(set(gain_dict.keys()).union(set(loss_dict.keys())))
                    key_list.sort(key=int, reverse=False)
                    for key in key_list:
                        fout.write(key)
                        fout.write('\t')
                        if key in gain_dict.keys():
                            fout.write(gain_dict[key])
                        else:
                            fout.write('0')
                        fout.write('\t')

                        if key in loss_dict.keys():
                            fout.write(loss_dict[key])
                        else:
                            fout.write('0')
                        fout.write('\n')

                # record the results in iTOL render file
                pos_gain_loss = dict()
                with open(number_output.format('itol_annotation').replace('tsv', 'txt'), 'w') as fin:
                    with open(itol_branch, 'r') as fout:
                        for line in fout:
                            fin.write(line)
                        fin.write('\n')
                    for key in gain_str_pf_list.keys():
                        if len(gain_str_pf_list[key]) > domain_limitation_value:
                            if key not in pos_tax.keys() and key != '0':
                                key_leaf = key + '0'
                                while key_leaf not in pos_tax.keys():
                                    key_leaf += '0'
                                key_right = key + '1'
                                while key_right not in pos_tax.keys():
                                    key_right += '1'
                                pos_gain_loss['{}|{}'.format(pos_tax[key_leaf], pos_tax[key_right])] = '+{}'.format(len(gain_str_pf_list[key]))
                            elif key in pos_tax.keys():
                                pos_gain_loss['{}'.format(pos_tax[key])] = '+{}'.format(len(gain_str_pf_list[key]))
                    for key in loss_str_pf_list.keys():
                        if len(loss_str_pf_list[key]) > domain_limitation_value:
                            if key not in pos_tax.keys() and key != '0':
                                key_leaf = key + '0'
                                while key_leaf not in pos_tax.keys():
                                    key_leaf += '0'
                                key_right = key + '1'
                                while key_right not in pos_tax.keys():
                                    key_right += '1'
                                key_tmp = '{}|{}'.format(pos_tax[key_leaf], pos_tax[key_right])
                                if key_tmp not in pos_gain_loss.keys():
                                    pos_gain_loss[key_tmp] = '-{}'.format(len(loss_str_pf_list[key]))
                                else:
                                    pos_gain_loss[key_tmp] = pos_gain_loss[key_tmp]+' -{}'.format(len(loss_str_pf_list[key]))
                            elif key in pos_tax.keys():
                                key_tmp = '{}'.format(pos_tax[key])
                                if key_tmp not in pos_gain_loss.keys():
                                    pos_gain_loss[key_tmp] = '-{}'.format(len(loss_str_pf_list[key]))
                                else:
                                    pos_gain_loss[key_tmp] = pos_gain_loss[key_tmp]+' -{}'.format(len(loss_str_pf_list[key]))
                    for key in pos_gain_loss.keys():
                        str_tmp = '{},{},0,#000000,normal,1,0\n'.format(key, pos_gain_loss[key])
                        fin.write(str_tmp)

            if if_overlaps:
                print('Collecting domain combinations')
                # 2.3 domains combinations
                for e_type in ['gain', 'loss']:
                    domain_file = domain_combination_output.format(e_type)
                    if e_type == 'gain' and not os.path.exists(domain_file):
                        relation_collection(gain_str_pf_list, domain_file)
                    elif e_type == 'loss' and not os.path.exists(domain_file):
                        relation_collection(loss_str_pf_list, domain_file)

                gain_and_loss_collection(domain_combination_output.format('gain'), domain_combination_output.format('loss'), domain_combination_output.format('gain_loss'))


if __name__ == '__main__':
    # The name of each strain should be like： 'tax_id' or tax_id.
    # => tax supports alphabet, digital, '-', '.', and '_'; id supports alphabet, digital, and '-' .
    base_path = ''
    # input files
    json_file = os.path.join(base_path, 'inputs', 'dict_pf_species.json')
    ref_tree = os.path.join(base_path, 'inputs', 'tree.newick')
    # templates of iTOL
    itol_node = os.path.join(base_path, 'inputs', 'itol_node_label.txt')
    itol_branch = os.path.join(base_path, 'inputs', 'itol_branch_label.txt')
    itol_branch_symbol = os.path.join(base_path, 'inputs', 'itol_branch_symbol.txt')
    domain_limitation_value = 0

    # middle results
    json_gain = os.path.join(base_path, 'middle_results', 'gain_pf_str.json')
    json_loss = os.path.join(base_path, 'middle_results', 'loss_pf_str.json')
    json_never = os.path.join(base_path, 'middle_results', 'never_pf_str.json')

    # domain for transfer network & itol & sankey outputs
    domain = 'PF05925.13'
    # outputs & switch
    if_transfer_network = True
    transfer_network_output = os.path.join(base_path, 'outputs', 'single_domain', 'transfer_network_{}.txt'.format(domain))

    if_itol_sankey = True
    itol_output = os.path.join(base_path, 'outputs', 'single_domain', 'itol_render_{}.txt'.format(domain))
    sankey_output = os.path.join(base_path, 'outputs', 'single_domain', 'sankey_phy_type_{}.tsv'.format(domain))

    if_number = True
    number_output = os.path.join(base_path, 'outputs', 'number_collection', 'number_collection_{}.tsv')

    if_overlaps = True
    domain_combination_output = os.path.join(base_path, 'outputs', 'domain_combination', 'domain_combination_{}.tsv')

    # ************************************** Protein domain *********************************************************
    # pf_dict['PF00627.32'] = [tax_id_1, tax_id_2, ...]
    pf_dict = pf_dict_load(json_file)
    # ************************************ Tree information *********************************************************
    # spe_tax[tax_id] = tax
    # name_position[tax_id] = ['1111']
    # short_distance_dict['1111'] = 0.22: The distance between '1111' and '111' is 0.22
    # long_distance_dict['1111'] = 0.74: The distance between '1111' and '0' is 0.74
    # pos_tax['1111'] = tax_id
    spe_tax = dict()
    name_position = defaultdict(list)
    long_distance_dict = dict()
    short_distance_dict = dict()
    pos_tax = dict()
    nj_tree_post_process(ref_tree, name_position, long_distance_dict, spe_tax, short_distance_dict)
    for name in name_position.keys():
        pos_tax[name_position[name]] = name

    # ******************************* Clustering & gain and loss event detection ************************************
    method_list = ['DBSCAN', 'MP', 'ML_NUM', 'ML_DIS', 'SD']
    method = method_list[1]
    clustering_and_detection(method, name_position, pf_dict, spe_tax, short_distance_dict, pos_tax, domain, json_gain, json_loss, json_never, if_transfer_network, if_itol_sankey, if_number, if_overlaps, transfer_network_output, itol_output, sankey_output, number_output, domain_combination_output, itol_node, itol_branch, domain_limitation_value, itol_branch_symbol)
    print('Done, please find results in the outputs document')
