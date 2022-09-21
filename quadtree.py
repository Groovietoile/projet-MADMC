import numpy as np

class Noeud(object):
    def __init__(self, solution, v, parent=None, k=None):
        self.solution = solution #indice des objets d'une solution
        self.v = np.array(v) #vecteur v des solutions avec la valeur pour chaque critère
        self.fils = [] #liste des noeuds fils
        self.parent = parent #le noeud parent
        self.k = np.array(k) #chaîne binaire pour le quadtree
        
    def reset_noeud(self):
        self.fils = []
        self.parent = None
        self.k = None

#implémentation du quadtree dans le cadre d'un problème de maximisation        
class QuadTree(object):

    def __init__(self, nb_criteres, racine=None, noeuds=[]):
        self.racine = racine
        self.noeuds = noeuds #solutions non dominées
        self.nb_criteres = nb_criteres

    def get_branche(self, racine_branche, liste_noeuds=set()):
        """ Récupération récursive de tous les noeuds fils de racine_branche + liste_noeuds

        Parametres
        ----------
        racine_branche : Noeud
        liste_noeuds : liste
        
        Returns
        -------
        liste_noeuds : liste
        """        
        liste_noeuds = liste_noeuds|set(racine_branche.fils)
        for fils in racine_branche.fils:
            if(fils.fils != []):
                liste_noeuds = liste_noeuds|self.get_branche(fils, liste_noeuds)
        return liste_noeuds

    #vide le Quad-tree
    def reset(self):
        self.racine = None
        self.noeuds = []
        for noeud in self.noeuds:
            noeud.reset_noeud()

    def noeud_est_domine_par_branche(self, noeud, k, racine_branche):
        """ Renvoie true si le noeud est dominé par les noeuds fils de racine_branche

        Parametres
        ----------
        noeud : Noeud
        k : chaîne binaire
        racine_branche : noeud

        Returns
        -------
        bool
        """ 

        if np.all(k >= racine_branche.k):
            if np.all(noeud.v <= racine_branche.v):
                return True
            else:
                for fils in racine_branche.fils:
                    k1 = np.array([0]*self.nb_criteres)
                    k1[np.where(noeud.v <= racine_branche.v)] = 1
                    if self.noeud_est_domine_par_branche(noeud, k1, fils):
                        return True
        return False

    def noeud_domine_branche(self, noeud, k, racine_branche):
        """ Renvoie true si le noeud domine les noeuds fils de racine_branche et les fils du noeud à réinsérer

        Parametres
        ----------
        noeud : Noeud
        k : chaîne binaire
        racine_branche : noeud

        Returns
        -------
        bool
        """ 
        if racine_branche not in self.noeuds:
            return False, None

        fils_noeud = []

        if np.all(racine_branche.k >= k):
            if np.all(racine_branche.v <= noeud.v):
                return True, [racine_branche]
            else:
                for fils in racine_branche.fils:
                    k1 = np.array([0]*self.nb_criteres)
                    k1[np.where(noeud.v <= racine_branche.v)] = 1
                    condition, fils_noeud_bis = self.noeud_domine_branche(noeud, k1, fils)
                    if condition:
                        fils_noeud = fils_noeud_bis

            if len(fils_noeud) != 0:
                return True, fils_noeud

        return False, None

    def inserer(self, noeud, racine_branche=None):
        """ Insère noeud dans l'arbre ayant pour racine racine_branche si il est non dominé
        Renvoie true si il a été inséré

        Parametres
        ----------
        noeud : Noeud à insérer
        racine_branche : noeud

        Returns
        -------
        bool
        """

        # Racine domine noeud
        if(np.all(noeud.k == 1)):
            return False

        # La solution noeud est déjà dans l'arbre
        for n in self.noeuds:
            if np.all(noeud.v == n.v):
                return False

        if(racine_branche == None):
            racine_branche = self.racine

        if(self.racine == None):
            self.racine = noeud
            return True

        # Initialisation chaîne binaire de noeud
        noeud.k = np.array([0]*self.nb_criteres)
        noeud.k[np.where(noeud.v <= racine_branche.v)] = 1

        # Racine dominée par noeud
        if(np.all(noeud.k == 0)):
            if(np.all(noeud.v == racine_branche.v)):
                return False

            # Remplacement racine
            self.racine = noeud
            noeuds_a_reinserer = []
            for n in self.noeuds:
                self.noeuds.remove(n)
                n.parent = None
                n.fils = []
                n.k = None
                noeuds_a_reinserer.append(n)

            for n in noeuds_a_reinserer:
                self.inserer(n)
            return True
        
        for fils in racine_branche.fils:
            if(self.noeud_est_domine_par_branche(noeud, noeud.k, fils)):
                return False

        noeuds_a_reinserer = []
        for fils in racine_branche.fils:
            is_domine, subtrees = self.noeud_domine_branche(noeud,noeud.k,fils)

            # Fils est domine par noeud
            if(is_domine):
                for subtree in subtrees:
                    if(subtree not in self.noeuds):
                        continue
                    self.noeuds.remove(subtree)
                    subtree.parent.fils.remove(subtree)
                    subtree.reset_noeud()
                    li=self.get_branche(subtree)

                    for s_fils in li:
                        s_fils.reset_noeud()
                        self.noeuds.remove(s_fils)
                        noeuds_a_reinserer.append(s_fils)
            
        for fils in racine_branche.fils:
            # Cas où la chaîne binaire est identique
            if(np.all(fils.k == noeud.k)):

                # On vérifie si noeud n'est pas dominé par fils, on l'insère
                if(np.all(noeud.v <= fils.v)):
                    for noeud in noeuds_a_reinserer:
                        self.inserer(noeud, self.racine)
                    return False

                if(self.inserer(noeud,fils)):
                    for noeud in noeuds_a_reinserer:
                        self.inserer(noeud, self.racine)
                    return True

        # Cas où on insère noeud s'il n'a pas de parent
        if(noeud.parent == None):
            noeud.parent = racine_branche
            racine_branche.fils.append(noeud)
            self.noeuds.append(noeud)

        # Cas où on insère noeud s'il a un noeud parent    
        else:
            if(noeud.parent != racine_branche):
                noeud.parent.fils.remove(noeud)
                noeud.parent = racine_branche
                racine_branche.fils.append(noeud)

        for i in range(len(noeuds_a_reinserer)):
            n = noeuds_a_reinserer[i]
            self.inserer(n,self.racine)

        return True