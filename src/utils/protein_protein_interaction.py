
class PPI():
    def __init__(self, ppp, interaction_confidence = -1, df_ecs = None, is_interacting = False, ap_ecs = None):
        # ppp   :   PPP :   protein-protein-pair object that interacts
        # interaction_confidence    :   float   :   confidence value if protein pair interacts (0: does not interact,
        #                                           1: interacts, -1: value not defined
        # is_interacting    :   boolean :   is this ppi actually interacting? only available if knowledge on labels are present
        # ap_ecs    :   [int]   :   direct reference to df_ecs: each 1 referres to an ap ec and each 0 means an actual negative ec
        # df_ecs    :   DataFrame   :   Dataframe containing all ECs that were considered with additional info (prediction_confidence)
        self.ppp = ppp
        self.interaction_confidence = interaction_confidence
        self.is_interacting = is_interacting
        self.df_ecs = df_ecs
        self.ap_ecs = ap_ecs


