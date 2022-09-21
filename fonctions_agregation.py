import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

def somme_criteres(v, objets):
	""" Evaluation d'une solution

	Parametres
	----------
	v : int
		nb de critères
	objets : liste

	Returns
	-------
	v_critères : liste
	""" 
	v_criteres = [0] * (len(objets[0])-1)
	for i in range(len(v)):
		if v[i] == 1:
			for j in range(len(v_criteres)):
				v_criteres[j] += objets[i][j+1]
	return v_criteres

def x_domine_y(x, y, w):
	""" Retourne True si x est préféré à y selon le décideur

	Parametres
	----------
	x : array
		Evaluation d'une solution x
	y : array
		Evaluation d'une solution y
	w : liste
		pondération
	Returns
	-------
	bool
	"""

	sp_x = SP(w, x)
	sp_y = SP(w, y)
	
	if(np.all(sp_x > sp_y)):
		return True
	return False

def SP(w, x):
	""" Calcule la somme pondérée

	Parametres
	----------
	w : liste
		pondération
	x : liste
		solution réalisable
	Returns
	-------
	sp : int
	"""
	sp = 0
	for i in range(len(w)):
		sp += x[i] * w[i]
	return sp

def PMR_SP(x, y, P = []):
	""" PMR appliqué à la somme pondérée

	Parametres
	----------
	x : array
		évaluation d'une solution s realisable
	y : array
		évaluation d'une solution s' realisable

	Returns
	-------
	var_values : array
	m.objVal : float
		max regret
	"""

	if len(x) != len(y):
		print ("erreur len(x)!=len(y)")

	# Supprime l'affichage
	env = Env(empty = True)
	env.setParam("OutputFlag", 0)
	env.start()

	# Création du modèle
	m = Model("PMR_SP", env=env)

	nbvar = len(x)
	colonnes = range(nbvar)

	# Contraintes 
	a = [1] * nbvar
	# Second membre
	b = 1.0
	# Parametre de fonction objectif
	c = (y-x)

		
 	# Déclaration variables de decision
	omega = []
	for i in colonnes:
		omega.append(m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="omega%d" % (i+1)))

	# Mise à jour du modèle pour intégrer les nouvelles variables
	m.update()

	obj = LinExpr();
	for i in range(nbvar):
		obj += c[i] * omega[i]

	# Definition de l'objectif, ici on maximise
	m.setObjective(obj,GRB.MAXIMIZE)

	# Definition des contraintes
	m.addConstr(quicksum(omega[j]*a[j] for j in colonnes) == b)
	for xi,yi in P:
		m.addConstr(quicksum(omega[j]*(xi-yi)[j] for j in colonnes) >= 0.0)

	#m.write('PMR_SP.lp')

	# Resolution
	m.optimize()

	# Problème infaisable
	if m.status == 3:
		return None, float("-inf")
		
	var_values = np.array([0.0] * nbvar)
	
	for j in colonnes:
		var_values[j] = omega[j].x
		
	return var_values, m.objVal
	
	
def MR(x, X, P):
	""" Minimax Regret

	Parametres
	----------
	x : array
		évaluation d'une solution
	X : list(array)
		evaluations de solutions

	Returns
	-------
	arg_mmr : array

	mmr : float

	"""
	arg_mr = np.array([0.0]*len(x))
	mr = float("-inf")
	for y in X:
		if np.all(x == y):
			continue
		
		w, res_pmr = PMR_SP(x, y, P)
		
		if res_pmr >= mr:
			arg_mr = y
			mr = res_pmr
			
	return arg_mr, mr
	
	
def MMR(X, P):
	""" Minimax Regret

	Parametres
	----------
	X: array
		evaluations de solutions

	Returns
	-------
	arg_mmr : array
		solution donnant le MMR
	mmr : float
		valeur du MMR
	"""
	arg_mmr = np.array([0.0] * len(X[0]))
	mmr = float("inf")
	
	for x in X:
		arg_mr, mr = MR(x, X, P)
		if mr <= mmr:
			arg_mmr = x
			mmr = mr
	
	return arg_mmr, mmr