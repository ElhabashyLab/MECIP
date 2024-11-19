import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.analysis_tools.skew_kurt import get_s_k_df
from scipy.stats import norm
from statistics import NormalDist
from src.analysis_tools.ecs import get_top_ecs, mark_true_ecs
from src.analysis_tools.contact_map import calculate_contact_map
from scipy.integrate import simps
from src.analysis_tools.solvent_access_ability import get_rsa_dfs
from matplotlib.colors import LogNorm
from src.analysis_tools.interface import read_interface_file


def plot_interface_size_vs_num_ap_ECs(ppps, params, filepath_out=None):
    # plots the interface size against the number of actual positive ECs for each complex
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath
    results = []
    if False:
        i=0
        for ppp in ppps:
            print(i)
            i+=1
            from src.analysis_tools.interface import read_interface_file, get_ap_interaction
            from src.analysis_tools.ecs import get_top_ecs, mark_true_ecs
            x = read_interface_file(ppp,params)
            if x is None:
                continue
            interface_size = len(x.index)
            seq_length = ppp.seq_length
            if seq_length is None or seq_length==0:
                continue
            rel_interface_size = interface_size/seq_length
            if interface_size>300: interface_size = 300
            df_top_ecs = get_top_ecs(ppp,params)
            df_top_ecs = mark_true_ecs(df_top_ecs, ppp.get_pdb_file(), params['TP_EC_distance_threshold_heavy_atoms'], ppp,
                                       params)
            df_true_ecs = df_top_ecs[df_top_ecs['true_ec']]
            ap = len(df_true_ecs.index)
            results.append([ppp.name, rel_interface_size, ap])

    results = [['sepmito_00434216', 7, 0], ['sepmito_00434210', 7, 0], ['sepmito_00473119', 57, 22], ['sepmito_00435060', 65, 8], ['sepmito_00434996', 78, 14], ['sepmito_00435007', 114, 9], ['sepmito_00435062', 28, 0], ['sepmito_00435059', 28, 0], ['sepmito_00822268', 27, 4], ['sepmito_00434907', 116, 51], ['sepmito_00913408', 28, 0], ['sepmito_00334055', 242, 34], ['sepmito_00334498', 55, 8], ['sepmito_00334371', 300, 24], ['sepmito_00050496', 8, 0], ['sepmito_00653256', 196, 0], ['sepmito_00472336', 89, 0], ['sepmito_00744981', 86, 2], ['sepmito_00422426', 17, 0], ['sepmito_00343962', 16, 2], ['sepmito_00304682', 29, 0], ['sepmito_00075562', 166, 78], ['sepmito_00963072', 114, 16], ['sepmito_00962910', 91, 3], ['sepmito_00963581', 114, 18], ['sepmito_00916065', 52, 2], ['sepmito_00916016', 129, 8], ['sepmito_00916049', 19, 2], ['sepmito_00915350', 34, 1], ['sepmito_00916019', 129, 1], ['sepmito_00472330', 89, 0], ['sepmito_00744975', 300, 4], ['sepmito_00422420', 17, 0], ['sepmito_00343956', 16, 2], ['sepmito_00304676', 29, 1], ['sepmito_00473181', 46, 10], ['sepmito_00473116', 54, 4], ['sepmito_00473058', 4, 0], ['sepmito_00473027', 45, 32], ['sepmito_00148964', 300, 42], ['sepmito_01003297', 22, 1], ['sepmito_00149067', 47, 16], ['sepmito_00867818', 43, 1], ['sepmito_00865293', 43, 0], ['sepmito_00075105', 166, 82], ['sepmito_00346217', 96, 22], ['sepmito_00745743', 15, 3], ['sepmito_00822370', 31, 9], ['sepmito_00745654', 77, 8], ['sepmito_00475293', 74, 2], ['sepmito_00781227', 39, 3], ['sepmito_00475302', 6, 0], ['sepmito_00040064', 300, 1], ['sepmito_00637375', 48, 0], ['sepmito_00634754', 48, 0], ['sepmito_00646572', 48, 1], ['sepmito_00644549', 48, 0], ['sepmito_00031055', 300, 2], ['sepmito_00064983', 37, 7], ['sepmito_00328879', 52, 2], ['sepmito_00287891', 92, 13], ['sepmito_00165582', 128, 20], ['sepmito_00611644', 93, 21], ['sepmito_00850638', 24, 5], ['sepmito_00487199', 69, 6], ['sepmito_00892263', 32, 1], ['sepmito_00486413', 22, 5], ['sepmito_01043977', 7, 0], ['sepmito_00965940', 89, 25], ['sepmito_00716075', 28, 1], ['sepmito_00179988', 88, 3], ['sepmito_00152068', 11, 4], ['sepmito_01003303', 11, 1], ['sepmito_00152168', 50, 15], ['sepmito_00626078', 87, 3], ['sepmito_00422788', 33, 7], ['sepmito_00273411', 216, 70], ['sepmito_00480110', 79, 42], ['sepmito_01177769', 71, 0], ['sepmito_00829143', 35, 0], ['sepmito_00937209', 27, 20], ['sepmito_00920773', 79, 0], ['sepmito_00119072', 194, 75], ['sepmito_01003232', 11, 1], ['sepmito_00867753', 39, 5], ['sepmito_00776927', 15, 4], ['sepmito_00871018', 3, 0], ['sepmito_00776923', 15, 1], ['sepmito_00776813', 37, 1], ['sepmito_00776522', 149, 10], ['sepmito_00225945', 63, 0], ['sepmito_00153881', 300, 74], ['sepmito_00329193', 86, 14], ['sepmito_00288291', 77, 26], ['sepmito_00296130', 16, 0], ['sepmito_00486789', 23, 5], ['sepmito_00296575', 10, 3], ['sepmito_00296530', 50, 11], ['sepmito_00296534', 112, 14], ['sepmito_00868526', 23, 0], ['sepmito_00790344', 109, 0], ['sepmito_00825548', 5, 0], ['sepmito_01253986', 5, 0], ['sepmito_00801554', 109, 0], ['sepmito_00275675', 216, 66], ['sepmito_00165759', 197, 38], ['sepmito_00611833', 2, 2], ['sepmito_01044166', 69, 3], ['sepmito_00905132', 126, 5], ['sepmito_00905350', 63, 0], ['sepmito_00905788', 127, 15], ['sepmito_00905492', 47, 4], ['sepmito_00905376', 12, 1], ['sepmito_00905984', 31, 0], ['sepmito_00905932', 59, 3], ['sepmito_01142804', 17, 4], ['sepmito_00868577', 106, 10], ['sepmito_01003949', 29, 6], ['sepmito_00971701', 27, 0], ['sepmito_00899136', 54, 12], ['sepmito_00897975', 16, 3], ['sepmito_00897973', 79, 3], ['sepmito_01047728', 160, 12], ['sepmito_00405830', 95, 0], ['sepmito_00780907', 19, 0], ['sepmito_00360962', 10, 11], ['sepmito_01107068', 38, 5], ['sepmito_00288322', 89, 26], ['sepmito_00612104', 7, 2], ['sepmito_00328655', 5, 2], ['sepmito_00851113', 41, 16], ['sepmito_00892738', 73, 6], ['sepmito_00486816', 146, 21], ['sepmito_00329083', 43, 25], ['sepmito_00191131', 8, 2], ['sepmito_00851061', 13, 0], ['sepmito_00892686', 123, 9], ['sepmito_00288177', 115, 58], ['sepmito_00288114', 15, 5], ['sepmito_00288120', 95, 20], ['sepmito_00611864', 24, 0], ['sepmito_00892484', 18, 2], ['sepmito_01044198', 90, 9], ['sepmito_00567252', 72, 0], ['sepmito_01394104', 50, 0], ['sepmito_00266853', 57, 9], ['sepmito_00632245', 39, 1], ['sepmito_01107341', 22, 4], ['sepmito_00965977', 7, 6], ['sepmito_00965941', 20, 7], ['sepmito_01112127', 8, 1], ['sepmito_01112111', 55, 3], ['sepmito_01111561', 26, 2], ['sepmito_00780922', 216, 0], ['sepmito_00781303', 45, 0], ['sepmito_00781383', 7, 1], ['sepmito_01107634', 18, 0], ['sepmito_00759535', 23, 4], ['sepmito_00945764', 132, 5], ['sepmito_01299297', 6, 0], ['sepmito_00766588', 23, 5], ['sepmito_00946114', 4, 0], ['sepmito_01314665', 6, 0], ['sepmito_01147150', 69, 8], ['sepmito_01037918', 4, 0], ['sepmito_00971719', 29, 6], ['sepmito_00971735', 76, 5], ['sepmito_00659321', 300, 6], ['sepmito_00545301', 107, 10], ['sepmito_00768792', 15, 4], ['sepmito_00545514', 32, 6], ['sepmito_00354602', 15, 0], ['sepmito_00545347', 29, 3], ['sepmito_00545539', 32, 0], ['sepmito_00596156', 300, 0], ['sepmito_00611416', 8, 1], ['sepmito_00612213', 17, 7], ['sepmito_01044733', 15, 0], ['sepmito_00611937', 43, 19], ['sepmito_00582530', 81, 14], ['sepmito_00761321', 64, 1], ['sepmito_01243854', 177, 8], ['sepmito_01106991', 10, 0], ['sepmito_00851913', 70, 7], ['sepmito_01133166', 32, 0], ['sepmito_01138143', 80, 8], ['sepmito_00105259', 83, 0], ['sepmito_00601647', 24, 1], ['sepmito_00313367', 29, 1], ['sepmito_00236020', 5, 0], ['sepmito_01243728', 30, 0], ['sepmito_00851787', 10, 0], ['sepmito_01141925', 8, 8], ['sepmito_00207142', 13, 12], ['sepmito_00168805', 5, 2], ['sepmito_00171776', 3, 0], ['sepmito_01212924', 300, 5], ['sepmito_00636521', 300, 0], ['sepmito_00636129', 65, 0], ['sepmito_00681917', 163, 7], ['sepmito_00626123', 24, 1], ['sepmito_00499951', 15, 0], ['sepmito_00405096', 15, 0], ['sepmito_00483669', 15, 0], ['sepmito_00940509', 88, 8], ['sepmito_00858058', 86, 12], ['sepmito_01051555', 30, 4], ['sepmito_00203819', 71, 1], ['sepmito_00871013', 9, 0], ['sepmito_00758933', 81, 7], ['sepmito_00759538', 21, 1], ['sepmito_01299184', 128, 7], ['sepmito_00759289', 106, 3], ['sepmito_01314549', 128, 4], ['sepmito_00851251', 24, 11], ['sepmito_00601864', 10, 0], ['sepmito_01142143', 22, 0], ['sepmito_00871019', 12, 2], ['sepmito_00871010', 12, 6], ['sepmito_00871016', 3, 1], ['sepmito_00871015', 9, 0], ['sepmito_00870781', 129, 3], ['sepmito_00870477', 113, 10], ['sepmito_00913586', 12, 6], ['sepmito_00922329', 5, 0], ['sepmito_01133513', 37, 5], ['sepmito_00893007', 27, 1], ['sepmito_01154443', 53, 5], ['sepmito_01146849', 24, 3], ['sepmito_01147128', 36, 5], ['sepmito_01145879', 60, 2], ['sepmito_00174126', 300, 0], ['sepmito_00173843', 300, 3], ['sepmito_00908095', 3, 0], ['sepmito_00908475', 35, 1], ['sepmito_00907077', 116, 9], ['sepmito_00921364', 85, 0], ['sepmito_00908600', 22, 2], ['sepmito_01244640', 25, 0], ['sepmito_01133953', 23, 1], ['sepmito_00269153', 133, 19], ['sepmito_00583453', 10, 2], ['sepmito_00602003', 107, 25], ['sepmito_00601890', 1, 0], ['sepmito_01152805', 56, 4], ['sepmito_00840067', 30, 5], ['sepmito_00581096', 57, 0], ['sepmito_01154425', 89, 18], ['sepmito_01154377', 28, 1], ['sepmito_01154428', 21, 1], ['sepmito_01153868', 14, 1], ['sepmito_00761756', 114, 1], ['sepmito_01244383', 21, 2], ['sepmito_00775785', 22, 0], ['sepmito_00582929', 48, 0], ['sepmito_00922492', 116, 7], ['sepmito_01107505', 39, 1], ['sepmito_00852388', 8, 0], ['sepmito_01133695', 96, 11], ['sepmito_00829040', 109, 5], ['sepmito_00921126', 5, 0], ['sepmito_00937125', 96, 13], ['sepmito_00599855', 39, 0], ['sepmito_00722216', 28, 1], ['sepmito_00737067', 3, 0], ['sepmito_00447425', 19, 1], ['sepmito_00447804', 37, 5], ['sepmito_00447716', 11, 2], ['sepmito_00167867', 202, 0], ['sepmito_00857945', 10, 0], ['sepmito_00459925', 1, 2], ['sepmito_00214796', 41, 0], ['sepmito_00137510', 56, 0], ['sepmito_00137140', 56, 0], ['sepmito_00922603', 1, 0], ['sepmito_01107623', 48, 3], ['sepmito_01133851', 8, 2], ['sepmito_00779743', 10, 1], ['sepmito_00354634', 24, 0], ['sepmito_00779227', 117, 10], ['sepmito_00791282', 76, 9], ['sepmito_00892878', 27, 3], ['sepmito_00892558', 119, 8], ['sepmito_00930840', 22, 1], ['sepmito_00892495', 2, 0], ['sepmito_01177809', 13, 0], ['sepmito_00632165', 20, 0], ['sepmito_01244088', 7, 0], ['sepmito_00858095', 6, 3], ['sepmito_01142285', 5, 6], ['sepmito_01152499', 25, 0], ['sepmito_00875028', 30, 5], ['sepmito_01107544', 107, 10], ['sepmito_01152832', 92, 11], ['sepmito_01027989', 39, 9], ['sepmito_01027998', 60, 20], ['sepmito_01026842', 124, 5], ['sepmito_00868535', 216, 22], ['sepmito_00868501', 73, 0], ['sepmito_00868518', 45, 4], ['sepmito_01254003', 45, 0], ['sepmito_00868506', 73, 0], ['sepmito_00744820', 10, 1], ['sepmito_00993925', 55, 7], ['sepmito_00354461', 152, 2], ['sepmito_00152814', 1, 0], ['sepmito_00913294', 10, 1], ['sepmito_00857979', 42, 6], ['sepmito_00203793', 33, 0], ['sepmito_00721819', 72, 0], ['sepmito_01142169', 36, 5], ['sepmito_00921316', 41, 0], ['sepmito_00829195', 144, 16], ['sepmito_00828729', 127, 17], ['sepmito_01244590', 19, 1], ['sepmito_01133181', 1, 0], ['sepmito_00354630', 24, 0], ['sepmito_00744590', 117, 11], ['sepmito_00934560', 3, 3], ['sepmito_00486638', 15, 1], ['sepmito_01037455', 26, 0], ['sepmito_00348923', 30, 0], ['sepmito_01037912', 3, 0], ['sepmito_00138237', 51, 0], ['sepmito_00825069', 60, 4], ['sepmito_00138226', 137, 34], ['sepmito_00627797', 39, 2], ['sepmito_01253167', 60, 0], ['sepmito_00367779', 35, 2], ['sepmito_01298428', 11, 1], ['sepmito_00526916', 107, 2], ['sepmito_00765973', 81, 6], ['sepmito_00354150', 300, 0], ['sepmito_00152718', 3, 0], ['sepmito_01313790', 11, 0], ['sepmito_00476969', 35, 0], ['sepmito_01036679', 130, 0], ['sepmito_01036677', 2, 1], ['sepmito_00874448', 57, 0], ['sepmito_00394426', 89, 4], ['sepmito_00775605', 162, 0], ['sepmito_00394506', 8, 0], ['sepmito_00717007', 33, 0], ['sepmito_00921379', 4, 0], ['sepmito_00920792', 9, 1], ['sepmito_01052168', 131, 43], ['sepmito_00936832', 127, 28], ['sepmito_01052180', 2, 0], ['sepmito_01051569', 53, 0], ['sepmito_00822150', 107, 5], ['sepmito_00865481', 8, 0], ['sepmito_00865908', 14, 4], ['sepmito_01254001', 14, 0], ['sepmito_00825546', 42, 0], ['sepmito_00293649', 23, 0], ['sepmito_01244649', 33, 4], ['sepmito_01134046', 30, 2], ['sepmito_01052123', 7, 1], ['sepmito_00857925', 21, 0], ['sepmito_00722334', 2, 0], ['sepmito_00768789', 15, 1], ['sepmito_00354632', 37, 0], ['sepmito_00768414', 149, 9], ['sepmito_00207509', 5, 2], ['sepmito_00916136', 24, 2], ['sepmito_00766592', 21, 2], ['sepmito_00993958', 93, 2], ['sepmito_00775305', 206, 10], ['sepmito_00775468', 33, 0], ['sepmito_00775985', 94, 6], ['sepmito_01152937', 2, 3], ['sepmito_00775885', 43, 0], ['sepmito_00930702', 8, 5], ['sepmito_00736442', 14, 3], ['sepmito_00736454', 57, 1], ['sepmito_00161588', 300, 14], ['sepmito_00947888', 50, 2], ['sepmito_01177798', 34, 0], ['sepmito_00500514', 11, 0], ['sepmito_01107664', 8, 0], ['sepmito_01153056', 24, 5], ['sepmito_00527121', 32, 0], ['sepmito_00354506', 51, 0], ['sepmito_00152821', 16, 0], ['sepmito_00929491', 39, 1], ['sepmito_01299190', 128, 4], ['sepmito_01298793', 27, 0], ['sepmito_01047375', 4, 1], ['sepmito_00825279', 1, 0], ['sepmito_01253380', 1, 0], ['sepmito_00354598', 15, 0], ['sepmito_00526962', 29, 1], ['sepmito_00527144', 32, 0], ['sepmito_00766330', 106, 1], ['sepmito_01314555', 128, 6], ['sepmito_00721765', 51, 0], ['sepmito_00825547', 42, 0], ['sepmito_00354203', 266, 43], ['sepmito_00913328', 24, 0], ['sepmito_01314155', 27, 2], ['sepmito_00354571', 51, 9], ['sepmito_00954435', 22, 0], ['sepmito_00001899', 96, 51], ['sepmito_00145377', 111, 30], ['sepmito_00088334', 46, 20], ['sepmito_01046534', 106, 2], ['sepmito_00349732', 59, 2], ['sepmito_01046532', 65, 0], ['sepmito_00001765', 27, 5], ['sepmito_00913022', 117, 8], ['sepmito_00477014', 16, 1], ['sepmito_01177132', 34, 0], ['sepmito_00172148', 201, 43], ['sepmito_00168902', 32, 8], ['sepmito_00168890', 161, 33], ['sepmito_01296045', 300, 0], ['sepmito_01048893', 300, 0], ['sepmito_00728993', 300, 0], ['sepmito_00875947', 17, 0], ['sepmito_00560453', 62, 0], ['sepmito_00881593', 80, 0], ['sepmito_00604695', 75, 0], ['sepmito_00536302', 300, 0], ['sepmito_01296070', 300, 0], ['sepmito_01296337', 300, 1], ['sepmito_00893293', 102, 0], ['sepmito_01365165', 82, 1], ['sepmito_00172046', 73, 15], ['sepmito_00209632', 128, 18], ['sepmito_00088433', 242, 41], ['sepmito_01048468', 79, 0], ['sepmito_00686408', 96, 0], ['sepmito_01365740', 96, 0], ['sepmito_00498973', 300, 0], ['sepmito_01330236', 106, 1], ['sepmito_01365237', 78, 0], ['sepmito_01330197', 106, 0], ['sepmito_01365171', 113, 2], ['sepmito_01365143', 94, 1], ['sepmito_00614183', 113, 1], ['sepmito_00567225', 72, 0], ['sepmito_01394077', 50, 0], ['sepmito_00964346', 49, 3], ['sepmito_00540269', 126, 0], ['sepmito_00539993', 173, 0], ['sepmito_00814046', 22, 0], ['sepmito_00734472', 159, 0], ['sepmito_01018010', 51, 0], ['sepmito_00796346', 117, 0], ['sepmito_00567223', 72, 0], ['sepmito_01394075', 300, 0], ['sepmito_00391033', 300, 0], ['sepmito_00996612', 50, 0], ['sepmito_00996622', 50, 0], ['sepmito_00996621', 50, 0], ['sepmito_00996626', 50, 0], ['sepmito_00708180', 51, 0], ['sepmito_00567615', 300, 0], ['sepmito_00567230', 72, 0], ['sepmito_00567670', 300, 0], ['sepmito_00567674', 300, 0], ['sepmito_01386160', 115, 0], ['sepmito_01394082', 50, 0], ['sepmito_00543771', 103, 0], ['sepmito_00230938', 2, 0], ['sepmito_00549372', 16, 0], ['sepmito_00720827', 35, 3], ['sepmito_00605489', 300, 4], ['sepmito_00543775', 103, 0], ['sepmito_00675535', 8, 0], ['sepmito_00312921', 4, 0], ['sepmito_00657045', 56, 15], ['sepmito_01032148', 15, 0], ['sepmito_01032186', 214, 12], ['sepmito_00754787', 10, 2], ['sepmito_00848870', 25, 1], ['sepmito_00250987', 17, 0], ['sepmito_00675565', 39, 2], ['sepmito_01042914', 300, 0], ['sepmito_01205884', 300, 0], ['sepmito_00282471', 214, 4], ['sepmito_01405237', 0, 0], ['sepmito_00659470', 0, 0], ['sepmito_00941925', 0, 0], ['sepmito_00901619', 0, 0], ['sepmito_00936144', 0, 0], ['sepmito_00947242', 0, 0], ['sepmito_01245759', 0, 0], ['sepmito_01401967', 0, 0], ['sepmito_00453933', 0, 0], ['sepmito_00717228', 0, 0], ['sepmito_00933671', 0, 0], ['sepmito_01245182', 0, 0], ['sepmito_00272727', 0, 0], ['sepmito_00642825', 0, 0], ['sepmito_00496321', 0, 0], ['sepmito_01328440', 0, 0], ['sepmito_01222992', 0, 0], ['sepmito_00776412', 0, 0], ['sepmito_01202073', 0, 0], ['sepmito_00069321', 0, 0], ['sepmito_00332714', 0, 0], ['sepmito_00158105', 0, 0], ['sepmito_01251049', 0, 0], ['sepmito_01011320', 0, 0], ['sepmito_00370437', 0, 0], ['sepmito_01103658', 0, 0], ['sepmito_00566367', 0, 0], ['sepmito_01080551', 0, 0], ['sepmito_00553435', 0, 0], ['sepmito_00526959', 0, 0], ['sepmito_01054556', 0, 0], ['sepmito_01339692', 0, 0], ['sepmito_00357378', 0, 0], ['sepmito_00046820', 0, 0], ['sepmito_00455867', 0, 0], ['sepmito_00462743', 0, 0], ['sepmito_00567701', 0, 0], ['sepmito_00438856', 0, 0], ['sepmito_01174472', 0, 0], ['sepmito_00321648', 0, 0], ['sepmito_00476705', 0, 0], ['sepmito_00635941', 0, 0], ['sepmito_00827443', 0, 0], ['sepmito_01236272', 0, 0], ['sepmito_00942706', 0, 0], ['sepmito_00948537', 0, 0], ['sepmito_00555113', 0, 0], ['sepmito_00360875', 0, 0], ['sepmito_01263608', 0, 0], ['sepmito_00324631', 0, 0], ['sepmito_00014626', 0, 0], ['sepmito_01056534', 0, 0], ['sepmito_01142434', 0, 0], ['sepmito_01141451', 0, 0], ['sepmito_01138815', 0, 0], ['sepmito_01149645', 0, 0], ['sepmito_00859724', 0, 0], ['sepmito_01271857', 0, 0], ['sepmito_00636187', 0, 0], ['sepmito_01238121', 0, 0], ['sepmito_01001188', 0, 0], ['sepmito_00266981', 0, 0], ['sepmito_00693911', 0, 0], ['sepmito_00261731', 0, 0], ['sepmito_00655057', 0, 0], ['sepmito_01054808', 0, 0], ['sepmito_00718366', 0, 0], ['sepmito_01017821', 0, 0], ['sepmito_00768135', 0, 0], ['sepmito_00754828', 0, 0], ['sepmito_00129996', 0, 0], ['sepmito_01240775', 0, 0], ['sepmito_00721085', 0, 0], ['sepmito_00598012', 0, 0], ['sepmito_01220623', 0, 0], ['sepmito_00526506', 0, 0], ['sepmito_00626098', 0, 0], ['sepmito_01086039', 0, 0], ['sepmito_01157580', 0, 0], ['sepmito_00337203', 0, 0], ['sepmito_00453156', 0, 0], ['sepmito_00592027', 0, 0], ['sepmito_01009259', 0, 0], ['sepmito_01161944', 0, 0], ['sepmito_00139858', 0, 0], ['sepmito_00176886', 0, 0], ['sepmito_01120311', 0, 0], ['sepmito_00571572', 0, 0], ['sepmito_00042839', 0, 0], ['sepmito_01252394', 0, 0], ['sepmito_01235186', 0, 0], ['sepmito_01053623', 0, 0], ['sepmito_01216128', 0, 0], ['sepmito_00766239', 0, 0], ['sepmito_00337402', 0, 0], ['sepmito_01344808', 0, 0], ['sepmito_00101771', 0, 0], ['sepmito_00806738', 0, 0], ['sepmito_01273713', 0, 0], ['sepmito_01343438', 0, 0], ['sepmito_01263199', 0, 0], ['sepmito_00322002', 0, 0], ['sepmito_01086303', 0, 0], ['sepmito_00311714', 0, 0], ['sepmito_00940256', 0, 0], ['sepmito_00458124', 0, 0], ['sepmito_01062929', 0, 0], ['sepmito_00245159', 0, 0], ['sepmito_01294110', 0, 0], ['sepmito_01032295', 0, 0], ['sepmito_01246183', 0, 0], ['sepmito_01054555', 0, 0], ['sepmito_00753482', 0, 0], ['sepmito_00806456', 0, 0], ['sepmito_00850667', 0, 0], ['sepmito_00656953', 0, 0], ['sepmito_00121372', 0, 0], ['sepmito_00498205', 0, 0], ['sepmito_00796611', 0, 0], ['sepmito_00911570', 0, 0], ['sepmito_01174389', 0, 0], ['sepmito_00658024', 0, 0], ['sepmito_00832041', 0, 0], ['sepmito_01278228', 0, 0], ['sepmito_00418416', 0, 0], ['sepmito_01281152', 0, 0], ['sepmito_00272910', 0, 0], ['sepmito_00588757', 0, 0], ['sepmito_00515669', 0, 0], ['sepmito_00543478', 0, 0], ['sepmito_01097749', 0, 0], ['sepmito_00975766', 0, 0], ['sepmito_01009216', 0, 0], ['sepmito_00245582', 0, 0], ['sepmito_00605408', 0, 0], ['sepmito_00468848', 0, 0], ['sepmito_00686055', 0, 0], ['sepmito_00673485', 0, 0], ['sepmito_00421234', 0, 0], ['sepmito_00483164', 0, 0], ['sepmito_00617980', 0, 0], ['sepmito_00537761', 0, 0], ['sepmito_00773411', 0, 0], ['sepmito_00135703', 0, 0], ['sepmito_00708249', 0, 0], ['sepmito_00513814', 0, 0], ['sepmito_00630295', 0, 0], ['sepmito_00090820', 0, 0], ['sepmito_00558486', 0, 0], ['sepmito_00335080', 0, 0], ['sepmito_01354331', 0, 0], ['sepmito_00336979', 0, 0], ['sepmito_00801877', 0, 0], ['sepmito_01277666', 0, 0], ['sepmito_00703528', 0, 0], ['sepmito_00793296', 0, 0], ['sepmito_01320239', 0, 0], ['sepmito_00477037', 0, 0], ['sepmito_01164198', 0, 0], ['sepmito_01283209', 0, 0], ['sepmito_00487743', 0, 0], ['sepmito_00379536', 0, 0], ['sepmito_01241377', 0, 0], ['sepmito_01144828', 0, 0], ['sepmito_00288624', 0, 0], ['sepmito_01104230', 0, 0], ['sepmito_00967563', 0, 0], ['sepmito_01285185', 0, 0], ['sepmito_00665871', 0, 0], ['sepmito_01273250', 0, 0], ['sepmito_00290798', 0, 0], ['sepmito_00886524', 0, 0], ['sepmito_00190705', 0, 0], ['sepmito_01153576', 0, 0], ['sepmito_00757854', 0, 0], ['sepmito_00035156', 0, 0], ['sepmito_00495903', 0, 0], ['sepmito_00887479', 0, 0], ['sepmito_00503319', 0, 0], ['sepmito_00807266', 0, 0], ['sepmito_01086850', 0, 0], ['sepmito_00251838', 0, 0], ['sepmito_00812770', 0, 0], ['sepmito_01224516', 0, 0], ['sepmito_00972092', 0, 0], ['sepmito_01049680', 0, 0], ['sepmito_00168150', 0, 0], ['sepmito_00839266', 0, 0], ['sepmito_00429801', 0, 0], ['sepmito_00531468', 0, 0], ['sepmito_01106631', 0, 0], ['sepmito_01170378', 0, 0], ['sepmito_01251902', 0, 0], ['sepmito_01246724', 0, 0], ['sepmito_01247649', 0, 0], ['sepmito_00370157', 0, 0], ['sepmito_01069105', 0, 0], ['sepmito_01336580', 0, 0], ['sepmito_01018959', 0, 0], ['sepmito_00420500', 0, 0], ['sepmito_01034294', 0, 0], ['sepmito_00960305', 0, 0], ['sepmito_00810455', 0, 0], ['sepmito_00135142', 0, 0], ['sepmito_01200701', 0, 0], ['sepmito_00242558', 0, 0], ['sepmito_01396258', 0, 0], ['sepmito_00474763', 0, 0], ['sepmito_01069883', 0, 0], ['sepmito_00279736', 0, 0], ['sepmito_00551002', 0, 0], ['sepmito_01331915', 0, 0], ['sepmito_01098505', 0, 0], ['sepmito_00261810', 0, 0], ['sepmito_00635699', 0, 0], ['sepmito_00253552', 0, 0], ['sepmito_00735308', 0, 0], ['sepmito_00460852', 0, 0], ['sepmito_00332980', 0, 0], ['sepmito_00859953', 0, 0], ['sepmito_00394861', 0, 0], ['sepmito_00787262', 0, 0], ['sepmito_00996568', 0, 0], ['sepmito_00958179', 0, 0], ['sepmito_00877310', 0, 0], ['sepmito_00157835', 0, 0], ['sepmito_00942189', 0, 0], ['sepmito_01344079', 0, 0], ['sepmito_00336878', 0, 0], ['sepmito_01401983', 0, 0], ['sepmito_00321686', 0, 0], ['sepmito_00588937', 0, 0], ['sepmito_00537845', 0, 0], ['sepmito_00912459', 0, 0], ['sepmito_00046868', 0, 0], ['sepmito_01235636', 0, 0], ['sepmito_01236672', 0, 0], ['sepmito_00370503', 0, 0], ['sepmito_00681295', 0, 0], ['sepmito_01330752', 0, 0], ['sepmito_00192122', 0, 0], ['sepmito_00822450', 0, 0], ['sepmito_00935907', 0, 0], ['sepmito_01238524', 0, 0], ['sepmito_01205957', 0, 0], ['sepmito_00760224', 0, 0], ['sepmito_01367419', 0, 0], ['sepmito_01162257', 0, 0], ['sepmito_00382513', 0, 0], ['sepmito_00005745', 0, 0], ['sepmito_00827906', 0, 0], ['sepmito_01382029', 0, 0], ['sepmito_00686228', 0, 0], ['sepmito_00371389', 0, 0], ['sepmito_01115185', 0, 0], ['sepmito_00650487', 0, 0], ['sepmito_00583328', 0, 0], ['sepmito_00086880', 0, 0], ['sepmito_00407911', 0, 0], ['sepmito_00574402', 0, 0], ['sepmito_00945430', 0, 0], ['sepmito_00447581', 0, 0], ['sepmito_00632966', 0, 0], ['sepmito_01269514', 0, 0], ['sepmito_00705582', 0, 0], ['sepmito_00020500', 0, 0], ['sepmito_00251899', 0, 0], ['sepmito_00640226', 0, 0], ['sepmito_00157938', 0, 0], ['sepmito_00924999', 0, 0], ['sepmito_00824895', 0, 0], ['sepmito_00880398', 0, 0], ['sepmito_01078870', 0, 0], ['sepmito_00717848', 0, 0], ['sepmito_00955403', 0, 0], ['sepmito_01277033', 0, 0], ['sepmito_00768206', 0, 0], ['sepmito_00397702', 0, 0], ['sepmito_01104119', 0, 0], ['sepmito_00689991', 0, 0], ['sepmito_00423878', 0, 0], ['sepmito_00251542', 0, 0], ['sepmito_00983546', 0, 0], ['sepmito_01011163', 0, 0], ['sepmito_01225505', 0, 0], ['sepmito_00957582', 0, 0], ['sepmito_00354732', 0, 0], ['sepmito_00992764', 0, 0], ['sepmito_01231397', 0, 0], ['sepmito_00811090', 0, 0], ['sepmito_01010973', 0, 0], ['sepmito_00452756', 0, 0], ['sepmito_01180991', 0, 0], ['sepmito_01285320', 0, 0], ['sepmito_00188603', 0, 0], ['sepmito_01064102', 0, 0], ['sepmito_01095221', 0, 0], ['sepmito_00370419', 0, 0], ['sepmito_01301872', 0, 0], ['sepmito_00217205', 0, 0], ['sepmito_01284306', 0, 0], ['sepmito_00818783', 0, 0], ['sepmito_00891427', 0, 0], ['sepmito_00844797', 0, 0], ['sepmito_00567582', 0, 0], ['sepmito_00205155', 0, 0], ['sepmito_00326549', 0, 0], ['sepmito_00269827', 0, 0], ['sepmito_00238192', 0, 0], ['sepmito_00009644', 0, 0], ['sepmito_00317569', 0, 0], ['sepmito_01262858', 0, 0], ['sepmito_00228923', 0, 0], ['sepmito_00316600', 0, 0], ['sepmito_01206456', 0, 0], ['sepmito_00673489', 0, 0], ['sepmito_00183175', 0, 0], ['sepmito_01331690', 0, 0], ['sepmito_00771270', 0, 0], ['sepmito_01057701', 0, 0], ['sepmito_00831556', 0, 0], ['sepmito_00215263', 0, 0], ['sepmito_00068274', 0, 0], ['sepmito_00660150', 0, 0], ['sepmito_00571108', 0, 0], ['sepmito_00406906', 0, 0], ['sepmito_01250484', 0, 0], ['sepmito_00296692', 0, 0], ['sepmito_01174381', 0, 0], ['sepmito_00653004', 0, 0], ['sepmito_00818069', 0, 0], ['sepmito_01238132', 0, 0], ['sepmito_01286938', 0, 0], ['sepmito_00494680', 0, 0], ['sepmito_00460271', 0, 0], ['sepmito_01237680', 0, 0], ['sepmito_01106351', 0, 0], ['sepmito_01332502', 0, 0], ['sepmito_01023275', 0, 0], ['sepmito_00540705', 0, 0], ['sepmito_00807548', 0, 0], ['sepmito_00431029', 0, 0], ['sepmito_00077681', 0, 0], ['sepmito_00389855', 0, 0], ['sepmito_00249700', 0, 0], ['sepmito_00177856', 0, 0], ['sepmito_00228511', 0, 0], ['sepmito_00273692', 0, 0], ['sepmito_00272973', 0, 0], ['sepmito_01090904', 0, 0], ['sepmito_01399703', 0, 0], ['sepmito_01219421', 0, 0], ['sepmito_01080236', 0, 0], ['sepmito_00945892', 0, 0], ['sepmito_00460296', 0, 0], ['sepmito_00624499', 0, 0], ['sepmito_01007975', 0, 0], ['sepmito_00519381', 0, 0], ['sepmito_00308145', 0, 0], ['sepmito_00441631', 0, 0], ['sepmito_00272978', 0, 0], ['sepmito_01172725', 0, 0], ['sepmito_00761070', 0, 0], ['sepmito_01396073', 0, 0], ['sepmito_00877080', 0, 0], ['sepmito_00390822', 0, 0], ['sepmito_00163997', 0, 0], ['sepmito_00954107', 0, 0], ['sepmito_00531733', 0, 0], ['sepmito_00375564', 0, 0], ['sepmito_01236594', 0, 0], ['sepmito_00776224', 0, 0], ['sepmito_00667500', 0, 0], ['sepmito_01122926', 0, 0], ['sepmito_00225014', 0, 0], ['sepmito_00494790', 0, 0], ['sepmito_01278256', 0, 0], ['sepmito_00461171', 0, 0], ['sepmito_01100291', 0, 0], ['sepmito_00709367', 0, 0], ['sepmito_00817330', 0, 0], ['sepmito_00134619', 0, 0], ['sepmito_01234832', 0, 0], ['sepmito_01386003', 0, 0], ['sepmito_00359454', 0, 0], ['sepmito_00941912', 0, 0], ['sepmito_01202553', 0, 0], ['sepmito_01360458', 0, 0], ['sepmito_00284663', 0, 0], ['sepmito_00892462', 0, 0], ['sepmito_01367144', 0, 0], ['sepmito_00918566', 0, 0], ['sepmito_00412539', 0, 0], ['sepmito_00295272', 0, 0], ['sepmito_00976879', 0, 0], ['sepmito_00111942', 0, 0], ['sepmito_01133507', 0, 0], ['sepmito_00958939', 0, 0], ['sepmito_00133029', 0, 0], ['sepmito_00812454', 0, 0], ['sepmito_01010912', 0, 0], ['sepmito_01348969', 0, 0], ['sepmito_00877973', 0, 0], ['sepmito_00946037', 0, 0], ['sepmito_00315358', 0, 0], ['sepmito_01269142', 0, 0], ['sepmito_00687317', 0, 0], ['sepmito_01092128', 0, 0], ['sepmito_01077551', 0, 0], ['sepmito_01016070', 0, 0], ['sepmito_01107530', 0, 0], ['sepmito_01325235', 0, 0], ['sepmito_00253791', 0, 0], ['sepmito_01085956', 0, 0], ['sepmito_00154489', 0, 0], ['sepmito_01175047', 0, 0], ['sepmito_01164103', 0, 0], ['sepmito_00907982', 0, 0], ['sepmito_00460320', 0, 0], ['sepmito_01024146', 0, 0], ['sepmito_01137240', 0, 0], ['sepmito_00002293', 0, 0], ['sepmito_01207554', 0, 0], ['sepmito_00967247', 0, 0], ['sepmito_00967508', 0, 0], ['sepmito_00252122', 0, 0], ['sepmito_00654635', 0, 0], ['sepmito_01331888', 0, 0], ['sepmito_01405715', 0, 0], ['sepmito_00656738', 0, 0], ['sepmito_01173567', 0, 0], ['sepmito_00039985', 0, 0], ['sepmito_00553936', 0, 0], ['sepmito_01323754', 0, 0], ['sepmito_00078869', 0, 0], ['sepmito_00417775', 0, 0], ['sepmito_00567570', 0, 0], ['sepmito_01263075', 0, 0], ['sepmito_00131465', 0, 0], ['sepmito_00995936', 0, 0], ['sepmito_01193010', 0, 0], ['sepmito_01024441', 0, 0], ['sepmito_01206921', 0, 0], ['sepmito_01119445', 0, 0], ['sepmito_00544982', 0, 0], ['sepmito_00596131', 0, 0], ['sepmito_00120348', 0, 0], ['sepmito_00941870', 0, 0], ['sepmito_00342793', 0, 0], ['sepmito_00316566', 0, 0], ['sepmito_00354790', 0, 0], ['sepmito_01247789', 0, 0], ['sepmito_01118786', 0, 0], ['sepmito_00831251', 0, 0], ['sepmito_00535108', 0, 0], ['sepmito_01313830', 0, 0], ['sepmito_01332209', 0, 0], ['sepmito_00623986', 0, 0], ['sepmito_00618395', 0, 0], ['sepmito_00877222', 0, 0], ['sepmito_00117164', 0, 0], ['sepmito_01273984', 0, 0], ['sepmito_01306370', 0, 0], ['sepmito_01117270', 0, 0], ['sepmito_00357650', 0, 0], ['sepmito_00125931', 0, 0], ['sepmito_01372074', 0, 0], ['sepmito_01011185', 0, 0], ['sepmito_01314845', 0, 0], ['sepmito_01186641', 0, 0], ['sepmito_00401653', 0, 0], ['sepmito_01210087', 0, 0], ['sepmito_00540780', 0, 0], ['sepmito_01064523', 0, 0], ['sepmito_01304167', 0, 0], ['sepmito_01291584', 0, 0], ['sepmito_00129403', 0, 0], ['sepmito_01229744', 0, 0], ['sepmito_00422518', 0, 0], ['sepmito_01290357', 0, 0], ['sepmito_00878152', 0, 0], ['sepmito_00338394', 0, 0], ['sepmito_00751735', 0, 0], ['sepmito_01342005', 0, 0], ['sepmito_00558955', 0, 0], ['sepmito_01026348', 0, 0], ['sepmito_00744288', 0, 0], ['sepmito_01130075', 0, 0], ['sepmito_01139672', 0, 0], ['sepmito_00924050', 0, 0], ['sepmito_00450409', 0, 0], ['sepmito_00630258', 0, 0], ['aaa0', 95.95891839553843, 37.98209210644392], ['aaa7', 23.426537058708647, 12.15324734756941], ['aaa10', 157.07923142309846, 42.55458010779634], ['aaa11', 6.517215989860642, 2.595132534795934], ['aaa12', 21.108884362604794, 10.560430503607119], ['aaa13', 139.3550819493753, 41.028510297697366], ['aaa17', 126.14345693246432, 18.574465649989396], ['aaa18', 21.9253374060689, 8.575394781981274], ['aaa19', 6.868605388761463, 5.142917408507918], ['aaa22', 109.73824923425514, 18.896743179972404], ['aaa26', 94.37089428635453, 23.707344474143248], ['aaa27', 179.75192634716103, 38.503255729597825], ['aaa32', 184.25881550644522, 56.45622765370253], ['aaa34', 33.632354564915424, 16.25027735125575], ['aaa37', 13.016121437375816, 15.055061846861571], ['aaa40', 22.237127950710438, 11.959883157327225], ['aaa41', 79.17814328263962, 16.498511167872866], ['aaa46', 168.6298803407507, 42.134481129072654], ['aaa47', 164.8097816009108, 41.737789512207414], ['aaa48', 142.58129273211935, 41.06895927858973], ['aaa50', 144.61020307232218, 34.4940901253753], ['aaa51', 37.7846790118176, 7.074110275693547], ['aaa52', 239.08048883730874, 61.41206303444106], ['aaa54', 45.58015303034935, 10.767135972631616], ['aaa56', 80.89781603372134, 32.95292652319307], ['aaa59', 121.4512560954408, 33.42842888445138], ['aaa62', 101.87416777330077, 24.546595968455737], ['aaa66', 122.31277617745545, 27.87810491483021], ['aaa67', 77.54091700675593, 26.28874752871485], ['aaa71', 21.89196490309637, 0], ['aaa74', 212.83381548210113, 45.200172022564445], ['aaa79', 108.23771027682342, 28.35309256226455], ['aaa80', 74.42436781334126, 17.311357896449802], ['aaa87', 0, 0], ['aaa88', 13.882026870856016, 4.357533761367481], ['aaa93', 231.11381612268704, 60.91428934837674], ['aaa94', 121.8325772534941, 45.85955333049234], ['aaa96', 244.26641696764509, 74.30361870582043], ['aaa98', 64.9525100128674, 23.410593910530974]]
    results = [x for x in results if x[1]<300]
    results = [[x[0],x[1]/600,x[2]] for x in results if x[1]<300]
    df_plot = pd.DataFrame(results, columns=['filename', 'rel_interface_size', 'num_ap_ecs']).set_index('filename')
    splot = sns.regplot(data=df_plot, x='rel_interface_size', y='num_ap_ecs', marker='+', ci=100, order=2, scatter_kws={'s':20})
    #splot.set(xscale="log", yscale="log")
    plt.title('regression of interface size and number of true ECs')
    # save or draw the plot
    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()
    exit()





def plot_kurtosis_vs_AP(ppis, params, filepath_out=None):
    # plots the kurtosis versus the number of actual positive ecs within the top considered section
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath
    results=[]
    for ppi in ppis:
        ppp = ppi.ppp
        kurtosis, skewness = ppp.calc_kurtosis_and_skewness(exclude_outside_pdb=params['remove_ecs_outside_of_monomer_pdb'])
        #df_top_ecs = get_top_ecs(ppp, params)
        #if not isinstance(df_top_ecs, pd.DataFrame): continue
        #df_top_ecs = mark_true_ecs(df_top_ecs, ppp.get_pdb_file(), params['TP_EC_distance_threshold_heavy_atoms'], ppp, params)
        #ap = sum(df_top_ecs['true_ec'])
        ap = sum(ppi.ap_ecs)
        results.append([ppp.name, kurtosis, skewness, ap])
    df_plot = pd.DataFrame(results, columns=['filename', 'excess_kurtosis', 'skewness', 'ap']).set_index('filename')
    splot = sns.scatterplot(data=df_plot, x='excess_kurtosis', y='ap', s=15)
    splot.set(xscale="log", yscale="log")
    plt.title('excess_kurtosis vs actual number of positive ECs')
    # save or draw the plot
    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()



def plot_kurtosis_vs_num_contacts(ppps, params, filepath_out=None):
    # plots the kurtosis versus the number of contacts
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath
    results=[]
    for ppp in ppps:
        if isinstance(ppp.computed_info_df, pd.DataFrame) and 'excess_kurtosis' in ppp.computed_info_df.columns:
            excess_kurtosis = float(ppp.computed_info_df['excess_kurtosis'])
        else:
            excess_kurtosis, skewness = ppp.calc_kurtosis_and_skewness(exclude_outside_pdb=params['remove_ecs_outside_of_monomer_pdb'])

        df_interface = read_interface_file(ppp, params)
        if isinstance(df_interface, pd.DataFrame): num_contacts = len(df_interface)
        else: num_contacts = 0
        # old code
        #dist_threshold = params['contact_map_heavy_atom_dist_threshold']
        #if isinstance(ppp.computed_info_df, pd.DataFrame) and f'num_contacts_heavy_atoms_{dist_threshold}A' in ppp.computed_info_df.columns:
        #    num_contacts = int(ppp.computed_info_df[f'num_contacts_heavy_atoms_{dist_threshold}A'])
        #else:
        #    _, heavy_atom_map = calculate_contact_map(ppp)
        #    if not isinstance(heavy_atom_map, np.ndarray): continue
        #    heavy_atom_map = [i for sublist in heavy_atom_map for i in sublist]
        #    num_contacts = sum(1 for x in heavy_atom_map if x>0 and x< dist_threshold)
        results.append([ppp.name, excess_kurtosis, num_contacts])
    df_plot = pd.DataFrame(results, columns=['filename', 'excess_kurtosis', 'num_contacts']).set_index('filename')
    splot = sns.scatterplot(data=df_plot, x='excess_kurtosis', y='num_contacts', s=15)
    splot.set(xscale="log", yscale="log")
    plt.title('excess_kurtosis vs number of contacts in given pdb file')
    # save or draw the plot
    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()


def plot_rsa_comparison_ap(ppis, params, filepath_out=None):
    # plots for all actual positive ecs the two rsa's compared in a scatterplot
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath
    max_range=200 if params['rsa_preferred_method'] == 'dssp' else 100
    results = [ [ 0 for i in range(max_range+1) ] for j in range(max_range+1) ]
    ap_ec_count = 0
    at_least_one_rsa_file_present = False
    for ppi in ppis:
        rsa_df_1 = get_rsa_dfs(ppi.ppp.protein1, params)
        rsa_df_2 = get_rsa_dfs(ppi.ppp.protein2, params)
        if rsa_df_1 is None or rsa_df_2 is None:
            continue
        at_least_one_rsa_file_present = True
        for ap_ec, (idx,row) in zip(ppi.ap_ecs, ppi.df_ecs.iterrows()):
            if ap_ec==1:
                try:
                    first_rsa = round(rsa_df_1.loc[row['i']][params['rsa_measurement']])
                    second_rsa = round(rsa_df_2.loc[row['j']][params['rsa_measurement']])
                except KeyError:
                    continue
                if first_rsa > max_range:
                    first_rsa = max_range
                if first_rsa < 0:
                    first_rsa = 0
                    continue
                if second_rsa > max_range:
                    second_rsa = max_range
                if second_rsa < 0:
                    second_rsa = 0
                    continue
                results[first_rsa][second_rsa] =results[first_rsa][second_rsa]+1
                ap_ec_count+=1
    if at_least_one_rsa_file_present:
        ax = sns.heatmap(results, cmap="YlGnBu", square=True, norm=LogNorm())
        plt.xlabel('rsa of res_1')
        plt.ylabel('rsa of res_2')
        plt.title(params['rsa_preferred_method'] + f' rsa comparison of {ap_ec_count} positive ECs from {len(ppis)} complexes')

        plt.tight_layout()
        if filepath_out:
            plt.savefig(filepath_out)
        else:
            plt.show()
        plt.clf()

def plot_rsa_comparison_an(ppis, params, filepath_out=None):
    # plots for all actual negatives ecs the two rsa's compared in a scatterplot
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath


    max_range=200 if params['rsa_preferred_method'] == 'dssp' else 100
    results = [ [ 0 for i in range(max_range+1) ] for j in range(max_range+1) ]
    an_ec_count = 0
    at_least_one_rsa_file_present=False
    for ppi in ppis:
        rsa_df_1 = get_rsa_dfs(ppi.ppp.protein1, params)
        rsa_df_2 = get_rsa_dfs(ppi.ppp.protein2, params)
        if rsa_df_1 is None or rsa_df_2 is None:
            continue
        at_least_one_rsa_file_present = True
        for ap_ec, (idx, row) in zip(ppi.ap_ecs, ppi.df_ecs.iterrows()):
            if ap_ec == 0:
                try:
                    first_rsa = round(rsa_df_1.loc[row['i']][params['rsa_measurement']])
                    second_rsa = round(rsa_df_2.loc[row['j']][params['rsa_measurement']])
                except KeyError:
                    continue
                if first_rsa > max_range:
                    first_rsa = max_range
                if first_rsa < 0:
                    first_rsa = 0
                    continue
                if second_rsa > max_range:
                    second_rsa = max_range
                if second_rsa < 0:
                    second_rsa = 0
                    continue
                results[first_rsa][second_rsa] = results[first_rsa][second_rsa] + 1
                an_ec_count += 1
    if at_least_one_rsa_file_present:
        ax = sns.heatmap(results, cmap="YlGnBu", square=True, norm=LogNorm())
        plt.xlabel('rsa of res_1')
        plt.ylabel('rsa of res_2')
        plt.title(params['rsa_preferred_method'] + f' rsa comparison of {an_ec_count} negative ECs from {len(ppis)} complexes')

        plt.tight_layout()
        if filepath_out:
            plt.savefig(filepath_out)
        else:
            plt.show()
        plt.clf()

#helper function:
def conservation(rel_i, conservation_df):
    #rel_i  :   int :   relative i: starts at 1 (not 0) and also is different for second monomer in a complex, since it keeps on counting (if first monomer is 100 res long, rel_i of the first residue of monomer 2 is 101)
    # reads and returns conservation value from dataframe
    try:
        return float(conservation_df[conservation_df['i'] == rel_i]['conservation'])
    except Exception as e:
        return None


def plot_conservation_vs_cn(ppis, params, filepath_out=None):
    # plots for each EC the conservation of either the single residue or both residues summed against the cn value of that ec
    # all dots are combined into one scatterplot
    # colors ap and an differently
    # filepath out is a list of 2 filepaths (one for added conservations one for single conservations)
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath
    results = []
    results2 = []
    for ppi in ppis:
        ppp = ppi.ppp
        df_top_ecs = ppi.df_ecs
        combined_length = ppp.seq_length

        if combined_length is None or ppp.protein1.get_sequence() is None or ppp.protein2.get_sequence() is None:
            continue
        frequencies_df = pd.read_csv(ppp.frequencies_file_in)
        first_length = len(ppp.protein1.get_sequence())
        second_length = len(ppp.protein2.get_sequence())

        if not frequencies_df['i'].max() == combined_length == first_length+ second_length:
            print(f'WARNING: something is wrong with the conservation file of {ppp.name} (most likely different residue count than sequences of concatinated protein pair)')

        # results = []
        for idx, row in df_top_ecs.iterrows():
            first_cons = conservation(row['i'], frequencies_df)
            second_cons = conservation(row['j'] + first_length, frequencies_df)
            if first_cons is None or second_cons is None:
                print(f'WARNING: something is wrong with the conservation file of {ppp.name} (most likely different residue count than sequences of concatinated protein pair)')
                continue
            total_cons = first_cons + second_cons
            cn = row['cn']
            ap = row['true_ec']
            results.append([total_cons, cn, ap])
            results2.append([first_cons, cn, ap])
            results2.append([second_cons, cn, ap])
    res_df = pd.DataFrame(results, columns=['total_conservation', 'cn', 'actual positive'])
    res_df2 = pd.DataFrame(results2, columns=['single_conservation', 'cn', 'actual positive'])


    for idx in range(2):
        if idx==0:
            sns.scatterplot(data=res_df, x='total_conservation', y='cn', hue='actual positive', s=2)
            plt.title(f'comparison of added conservation values against the cn score of {len(results)} ECs')
        if idx==1:
            sns.scatterplot(data=res_df2, x='single_conservation', y='cn', hue='actual positive', s=2)
            plt.title(f'comparison of conservation values of {len(results2)} single conservation values against the cn score of {len(results)} ECs')
        plt.tight_layout()
        if filepath_out:
            plt.savefig(filepath_out[idx])
        else:
            plt.show()
        plt.clf()

def plot_conservation_vs_cn_single_ppi(ppi, params, filepath_out=None):
    # plots for each EC the conservation of either the single residue or both residues summed against the cn value of that ec
    # each ppi gets its own plot, therefore this is only done for a single ppi
    # colors ap and an differently
    # filepath out is a list of 2 filepaths (one for added conservations one for single conservations)
    # if filepath_out is None it will be shown in the IDE, otherwise it will be saved at the filepath
    results = []
    results2 = []
    ppp = ppi.ppp
    df_top_ecs = ppi.df_ecs
    combined_length = ppp.seq_length

    if combined_length is None or ppp.protein1.get_sequence() is None or ppp.protein2.get_sequence() is None:
        return
    frequencies_df = pd.read_csv(ppp.frequencies_file_in)
    first_length = len(ppp.protein1.get_sequence())
    second_length = len(ppp.protein2.get_sequence())

    # results = []
    for idx, row in df_top_ecs.iterrows():
        first_cons = conservation(row['i'], frequencies_df)
        second_cons = conservation(row['j'] + first_length, frequencies_df)
        total_cons = first_cons + second_cons
        cn = row['cn']
        ap = row['true_ec']
        results.append([total_cons, cn, ap])
        results2.append([first_cons, cn, ap])
        results2.append([second_cons, cn, ap])
    res_df = pd.DataFrame(results, columns=['total_conservation', 'cn', 'actual positive'])
    res_df2 = pd.DataFrame(results2, columns=['single_conservation', 'cn', 'actual positive'])


    for idx in range(2):
        if idx==0:
            sns.scatterplot(data=res_df, x='total_conservation', y='cn', hue='actual positive')
            plt.title(f'comparison of added conservation values against the cn score of {len(results)} ECs')
        if idx==1:
            sns.scatterplot(data=res_df2, x='single_conservation', y='cn', hue='actual positive')
            plt.title(f'comparison of conservation values of {len(results2)} single conservation values against the cn score of {len(results)} ECs')
        plt.tight_layout()
        if filepath_out:
            plt.savefig(filepath_out[idx])
        else:
            plt.show()
        plt.clf()



def plot_pairplot(features,labels, feature_names, params, filepath_out=None):
    combined = [feature + [label] for (feature, label) in zip(features, labels)]
    df_plot = pd.DataFrame(combined, columns=feature_names + ['labels'])
    sns.pairplot(df_plot, hue='labels', height=2.5, palette='muted')


    plt.tight_layout()
    if filepath_out:
        plt.savefig(filepath_out)
    else:
        plt.show()
    plt.clf()
