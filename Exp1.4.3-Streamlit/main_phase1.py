#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:10:50 2022

Fusion de training.py et eval.py.
Qu'est-ce qui pourrait mal tourner ?

@author: Corentin.Boidot
"""

import streamlit as st
import numpy as np
import pandas as pd

import pickle
import time

from functools import reduce


st.set_page_config(layout="wide") # enfin !

pd.set_option("display.max_columns",40)
pd.set_option("display.max_rows",30)


USER = 7

#### Parameters

path = "" #"../Exp1-Streamlit/"


if "completed" not in st.session_state:
    st.session_state["completed"] = [False]*5

if "cur_page" not in st.session_state:
    st.session_state["cur_page"] = 0

if "cur_match" not in st.session_state:
    st.session_state["cur_match"] = 0

# i = st.session_state.cur_match -4

if "reponses_num" not in st.session_state:
    st.session_state["reponses_num"] = pd.DataFrame(np.zeros((0,3)),
                            columns=["rep","time_print","time_click"])
    
if "click_list" not in st.session_state:
    st.session_state["click_list"] = []    

if "t_print" not in st.session_state:
    st.session_state["t_print"] = None

clickable = False # clickable = True en dernière ligne !

## Format réponses

options_ope = ["l’équipe bleue va certainement gagner",
            "l’équipe bleue va vraisemblablement gagner",
            "l’équipe bleue a un léger avantage sur l’équipe rouge",
            "je ne sais pas",
            "l’équipe rouge a un léger avantage sur l’équipe bleue",
            "l’équipe rouge va vraisemblablement gagner",
            "l’équipe rouge va certainement gagner"]

interp_ope = {"l’équipe bleue va certainement gagner":0,
            "l’équipe bleue va vraisemblablement gagner":0,
            "l’équipe bleue a un léger avantage sur l’équipe rouge":0,
            "l’équipe rouge a un léger avantage sur l’équipe bleue":1,
            "l’équipe rouge va vraisemblablement gagner":1,
            "l’équipe rouge va certainement gagner":1}

temp_jeu = ["Il se passe rarement plus de quelques jours sans que je ne joue.",
                "Je joue au moins une fois par mois dernièrement.",
                "Il arrive facilement qu'un mois passe sans que je ne joue.",
                "J'ai fait une pause d'un an ou plus dernièrement."]

temp_visio = ["Il se passe rarement plus de quelques jours sans que je ne regarde un match.",
                "J'en regarde au moins une fois par mois dernièrement.",
                "Il arrive facilement qu'un mois passe sans que je ne ne regarde de match.",
                "J'ai fait une pause d'un an ou plus dernièrement."]


rangs = ["Non classé", "Fer", "Bronze", "Argent", "Or", "Platine", "Diamant",
         "Maître", "Grand Maître", "Challenger"]

datascience = ["aucune", "des bases en statistiques", "une formation math-info",
               "je travaille dans la data science",
               "expert en ML, avec des connaissances en XAI"]

r1 = None
r2 = None
r3 = None


old_order = ['firstBlood', 'blueDragons', 'blueHeralds', 'blueTowersDestroyed',
       'redDragons', 'redHeralds', 'redTowersDestroyed', 'blueWardsPlaced',
       'blueWardsDestroyed', 'blueKills', 'blueAssists', 'blueTotalGold',
       'blueTotalExperience', 'blueTotalMinionsKilled',
       'blueTotalJungleMinionsKilled', 'redWardsPlaced', 'redWardsDestroyed',
       'redKills', 'redAssists', 'redTotalGold', 'redTotalExperience',
       'redTotalMinionsKilled', 'redTotalJungleMinionsKilled']

new_order = ['firstBlood', 'blueDragons', 'redDragons', 'blueHeralds',
 'redHeralds', 'blueTowersDestroyed', 'redTowersDestroyed', 'blueWardsPlaced',
 'redWardsPlaced', 'blueWardsDestroyed', 'redWardsDestroyed', 'blueKills', 
 'redKills', 'blueAssists', 'redAssists', 'blueTotalGold', 'redTotalGold',
 'blueTotalExperience', 'redTotalExperience', 'blueTotalMinionsKilled', 
 'redTotalMinionsKilled', 'blueTotalJungleMinionsKilled',
 'redTotalJungleMinionsKilled']


nondutout = ["Non pas du tout", "non", "plutôt non", "indécis", "plutôt oui", "oui", "oui tout à fait"]
pasdaccord = ["Pas du tout d’accord", "pas d’accord", "plutôt pas d’accord", "neutre", "plutôt d’accord", "d’accord", "tout à fait d’accord"]

### Chargement des données ###


@st.cache
def init_data_tr():
    '''
    df_select : pd.DataFrame, passé au new_order
    '''

    # with open(path+"means.pkl",'rb') as pf:
    #     m = pickle.load(pf)
    #     m = m.apply(lambda x: round(x,2) if str(x)!=x else x)
    # df_select = xpo.data  # remplaçait les lignes suivantes
    with open(path+"selection_tr.pkl",'rb') as pf: 
        df_select = pickle.load(pf)
    # df_select.loc["moy"] = m
    # df_select.loc['moy','firstBlood'] = m.firstBlood
    df_select = df_select[new_order]
    with open(path+"label_tr.pkl",'rb') as f:
        labels = pickle.load(f)
        # 1 = victoire des rouges (ne pas se fier au nom de colonne, a)
    ### Préprocessing
    # on va randomiser n fonction de l'utilisateur
    # mais chaque bloc de 5 doit contenir au moins 1 rouge et un bleu.
    final_index = []
    blue_draw = []
    red_draw = []
    penury = 0 # 1 = rouge, -1 = bleu
    
    # random.seed(USER)
    
    labelshuffled = labels.sample(frac=1,random_state=USER)
    
    for i in range(30):
        cur_indx = labelshuffled.index[i]
        if penury==0: # cas normaux
            if len(final_index)%5 != 4:
                final_index.append(cur_indx) # inser
                if labelshuffled.iloc[i]:
                    red_draw.append(cur_indx)
                else:
                    blue_draw.append(cur_indx)
            else: # à la fin d'un paquet de 5, on contrôle, et on déclenche la
                # pénurie si nécessaire
                if blue_draw==[] :
                    penury = -1
                    red_draw = [cur_indx]
                    # print("blue penury")
                elif red_draw==[]:
                    penury = 1
                    blue_draw = [cur_indx]
                    # print("red penury")
                else:
                    final_index.append(cur_indx) #inser
                    blue_draw = []
                    red_draw = []
                    # print("bloc ok")
        elif penury==1: # on manque de rouge
            if not labelshuffled.iloc[i]:
                blue_draw.append(cur_indx) # on empile les bleus en trop
            else:
                final_index.append(cur_indx) # inser
                # le précédent bloc est donc fini, gestion de la pile now
                if len(blue_draw)<5: # fin de crise
                    final_index += blue_draw # multi-inser
                    penury = 0
                    # print("penury ends")
                else: # on prépare le prochain bloc qui attendra son rouge
                    final_index += blue_draw[:4] # multi-inser
                    blue_draw = blue_draw[4:] # la pile est réduite
        elif penury==-1: # on manque de bleu
            if labelshuffled.iloc[i]:
                red_draw.append(cur_indx) # on empile les bleus en trop
            else:
                final_index.append(cur_indx) # inser
                # le précédent bloc est donc fini, gestion de la pile now
                if len(red_draw)<5: # fin de crise
                    final_index += red_draw # multi-inser
                    penury = 0
                    # print("penury ends")
                else: # on prépare le prochain bloc qui attendra son bleu
                    final_index += red_draw[:4] # multi-inser
                    red_draw = red_draw[4:] # réduction de la pile
            # print(red_draw)
    labels = labels.loc[final_index]
    m = df_select.loc["moy"]
    df_select = df_select.loc[final_index]
    df_select.loc["moy"] = m
    
    with open(path+"results/shuffle"+str(USER)+".pkl",'wb') as f:
        pickle.dump(labels,f)
    
    labels = labels.reset_index(drop=True)
    # sans quoi, je ne peux les comparer pour donner une note à l'utilisateur
    return df_select, labels
df_tr, labels = init_data_tr()

@st.cache
def init_data_ev():
    '''
    df_select : pd.DataFrame, passé au new_order
    '''
    N = 40 # index à partir duquel on insère 
    with open(path+"selection_ev.pkl",'rb') as pf: 
        df_select = pickle.load(pf)

    df_select = df_select[new_order]
### traitement de la répétition
    # en maquette, je vais les répéter dans la deuxième dizaine.
    to_insert = df_select.iloc[:10].sample(5,random_state=USER)
    with open(path+"results/redondance_"+str(USER)+".pkl",'wb') as f:
        pickle.dump(to_insert,f)
    df_bis = df_select.iloc[:N].reset_index(drop=True)
    for i in range(5):
        df_bis.loc[N+2*i] = df_select.iloc[10+i] 
        df_bis.loc[N+2*i+1] = to_insert.iloc[i]
    df_bis = pd.concat([df_bis,df_select.reset_index(drop=True).iloc[N+5:50]]) \
        .reset_index(drop=True)
    df_bis.loc['moy'] = df_select.loc['moy']
    return df_bis
df_ev = init_data_ev()



### fonctions associées aux boutons "next" des différentes pages ###

def nextpage():
    global clickable
    if not clickable:
        # print("Ne cliquez pas si vite !")
        time.sleep(.8)
    else:
        if st.session_state.cur_page in [9,10]:
            k = st.session_state.reponses_form.shape[0]
            st.session_state['reponses_form'].loc[k] = r1
            st.session_state['reponses_form'].loc[k+1] = r2
            st.session_state['reponses_form'].loc[k+2] = r3
        elif st.session_state.cur_page in [8]:
            st.session_state['reponses_form'] = \
                pd.DataFrame(np.array(r1).T,columns=["rep"])
        st.session_state.cur_page += 1
        st.session_state.click_list.append(time.time())        
        st.session_state['t_print'] = None
        st.session_state["completed"] = [False]*5
        clickable = False
        time.sleep(.25)
        
def nextexample():
    global clickable
    if not clickable:
        # print("On ne vous a pas appris que le double-click, c'est mal ?")
        time.sleep(.8)
    else:
        st.session_state.cur_match += 1
        # print(r1)
        line = [r1, st.session_state.t_print, time.time()]
        st.session_state['reponses_num'].loc[st.session_state.cur_match-1] = line
        st.session_state['t_print'] = None
        clickable = False
        # st.session_state["slide"] = "je ne sais pas"
        c1 = st.session_state.cur_match >= 55
        c2 = st.session_state.cur_match%5==0 and st.session_state.cur_page<5
        if c1 or c2 :
            clickable = True
            nextpage()
        if st.session_state.cur_page==6:
            st.session_state.cur_page += 1
        time.sleep(.25)
            
    
def backtothree(three=3):
    global clickable
    if not clickable:
        # print("Ne cliquez pas si vite !")
        time.sleep(.8)
    else:
        clickable = False
        st.session_state.cur_page = three
        st.session_state.click_list.append(time.time())        
        st.session_state['t_print'] = None
        st.session_state["completed"] = [False]*5
        time.sleep(.25)
    
j = st.session_state.cur_page

# 'recense le premier mort de la partie (1 s’il est de l’équipe bleue, 0 s’il est de l’équipe rouge).'
@st.cache
def make_df_pres():
    l_ter = ["1e équipe à avoir infligé un kill ('blue' = bleue, 'red' = rouge).",
     'nombre de dragons (grand monstre) détruits par l’équipe bleue.',
     'nombre de héraults (grand monstre) détruits par l’équipe bleue.',
     'nombre de tourelles détruites par l’équipe bleue.',
     'nombre de dragons (grand monstre) détruits par l’équipe rouge.',
     'nombre de héraults (grand monstre) détruits par l’équipe rouge.',
     'nombre de tourelles détruites par l’équipe rouge.',
     'nombre de balises de vision placées par l’équipe bleue.',
     'nombre de balises ennemies détruites par l’équipe bleue.',
     'nombre de champions adverses tués par les champions bleus.',
     'nombre d’assists (coopérations de champions à un kill) au sein de l’équipe bleue.',
     'quantité totale de pièces d’or des champions bleus.',
     'quantité totale d’expérience des champions bleus.',
     'nombre de sbires ennemis détruits par les chamions bleus.',
     'nombre de petits monstres de la jungle détruits par les champions bleus.',
     'nombre de balises de vision placées par l’équipe rouge.',
     'nombre de balises ennemies détruites par l’équipe rouge.',
     'nombre de champions adverses tués par les champions rouges.',
     'nombre d’assists (coopérations de champions à un kill) au sein de l’équipe rouge.',
     'quantité totale de pièces d’or des champions rouges.',
     'quantité totale d’expérience des champions rouges.',
     'nombre de sbires ennemis détruits par les chamions rouges.',
     'nombre de petits monstres de la jungle détruits par les champions rouges.']
    keys = old_order
    dic = {"features_names":keys,"description":l_ter}
    df_pres = pd.DataFrame(dic)
    df_pres = df_pres.set_index("features_names")
    df_pres = df_pres.T[new_order].T #cassage de crâne
    return df_pres
df_pres = make_df_pres()



##############################################################################
#                              DÉBUT DE L'AFFICHAGE                          #
##############################################################################



if st.session_state.cur_page == 0:
    
    ### Présentation du problème
    
    st.title("Présentation de votre mission")
    "***À lire impérativement par tous.***\n"
    
    "On tente de faire des pronostics de League of Legends (prédire quelle équipe va à gagner). On prend les paris à partir de statistiques de jeu prises 10 minutes après le départ. Il s’agit de matchs classés solo de rang Diamant I à Master, durant la saison 10 de LoL (les joueurs ne se connaissent pas avant le début du match : les équipes sont formées de façon impartiales)."
    "Le programme que vous venez de lancer a pour but de fournir un entraînement à cette tâche."
    "Vous aurez une suite de match à évaluer, sous la forme de relevés de données de jeu. Votre évaluation revient à une décision (une sorte de pari) que vous seriez prêt à prendre vis-à-vis du match."
    # "Le format des données est présenté à la page suivante."
    "Pour vous préparer, le programme vous propose d'abord d'observer des données de matchs, sans évaluation."
    "Cliquez sur START pour commencer la préparation."
    
    st.button("START>>>",key='nekst',on_click=nextpage)

elif st.session_state.cur_page == 1:
    
    # i = st.session_state["cur_match"] - 4
    left_column, right_column = st.columns(2) # ou 3 ?

    with right_column:
        i = 0
       
        # "Pour prendre en main l'expérience, nous vous proposons de regarder les données de 5 matchs. Après quoi, vous aurez un entraînement rapide avant de commencer la tâche."
        "Regardez les données des 5 matchs avant de passer à la suite."
        i = st.selectbox("Choisissez le match à afficher", [1,2,3,4,5])
        vus = st.session_state.completed
        vus[i-1] = True
        
        winner = 'rouge' if labels.iloc[i-1] else 'bleue'
        st.title("Équipe gagnante : " + winner)
        
        # st.write(st.session_state.completed) # TO DO : effacer ça
        if reduce(lambda x,y: x and y, vus):
            increment = st.button("NEXT>>>",on_click=nextpage)
    
    tab = df_tr.iloc[[i-1,-1]].T

    tab.columns = ["match","moy"]
    left_column.title("Match d'observation n°"+str(i))
    left_column.table(tab.astype(str)) 
    
    st.sidebar.table(df_pres)
    


elif st.session_state.cur_page == 2:
    
    st.title("Transition")
    
    "À présent, vous allez devoir exprimer votre pronostic sur 5 matchs consécutivement. Suite à cela, vous pourrez voir vos potentielles erreurs."
    "Vous pourrez exprimer votre réponse sur une échelle d'incertitude/certitude en faveur d'une équipe ou de l'autre."
    "Suite à cette première évaluation, soit vous continuerez automatiquement l'entraînement, soit on vous proposera de passer à la tâche réelle."
    
    st.session_state.cur_match = 5
    
    st.button("NEXT>>>",on_click=nextpage)
    
elif st.session_state.cur_page == 3:
    
    left_column, right_column = st.columns(2) # ou 3 ?
    
    
    with right_column:
        
        i = st.session_state.cur_match -4
        st.title("Match n°"+str(i))
        
        title = 'Quel est votre pronostic ?'
        # "Pour prendre en main l'expérience, nous vous proposons de regarder les données de 5 matchs. Après quoi, vous aurez un entraînement rapide avant de commencer la tâche."
        vus = st.session_state.completed
        vus[(i-1)%5] = True
        
        r1 = st.select_slider(title,options_ope,"je ne sais pas",key="slide")
        
        # if reduce(lambda x,y: x and y, vus):
        #     increment = st.button("NEXT>>>",on_click=nextpage)
        # else:
        increment = st.button("NEXT>>>",on_click=nextexample)
    
    tab = df_tr.iloc[[i+4,-1]].T
    tab.columns = ["match","moy"]
    left_column.table(tab.astype(str)) # .style.format("{:.2f}") # table # dataframe
    
    st.sidebar.table(df_pres)
    

elif st.session_state.cur_page == 4:
    
    left_column, right_column = st.columns(2) # ou 3 ?
    # score computation
    i = st.session_state.cur_match -10
    extrait = st.session_state.reponses_num.loc[i+5:i+9,'rep']
    exprimes = extrait.apply(lambda x: False if x=="je ne sais pas" else True)
    numerateur = exprimes.sum()
    choix = extrait[exprimes].apply(lambda x: interp_ope[x])
    # labels.loc[i+5:i+9][exprimes]
    dividende = (choix.values == \
                 labels.loc[i+5:i+9][exprimes].values).sum()
    
    isok = dividende>=3 and numerateur-dividende<=1
    if "isok" not in st.session_state:
        st.session_state['isok'] = isok
    else:
        st.session_state['isok'] = isok or st.session_state.isok
    
    with right_column:
        
        st.title('Votre score : ' + str(dividende)+" / "+str(numerateur))
        # /!\ je réutilise i à une autre fin
        i = st.selectbox("Choisissez le match à afficher",
                         [i+1,i+2,i+3,i+4,i+5])
        vus = st.session_state.completed
        vus[(i-1)%5] = True
        
        winner = 'rouge' if labels.iloc[i+4] else 'bleue'
        st.title("Équipe gagnante : " + winner)
        predi = st.session_state.reponses_num.loc[i+4,'rep']
        st.title("Votre pronostic : " + predi)
        
        if st.session_state.cur_match >= 29:
            "Plus de match disponibles."
            st.button("Finir>>>",on_click=nextpage)
        else:
            if not isok:
                "Vous pouvez mieux faire."
                "Regardez vos erreurs et recommencez."
            if reduce(lambda x,y: x and y, vus):
                st.button("Recommencer>>>",on_click=backtothree)
            if isok:
                "Bien joué ! Vous pouvez passer à la suite si vous voulez, sinon vous pouvez refaire une série."
                st.button("Recommencer>>>",on_click=backtothree)
            if st.session_state.isok:
                st.button("Finir>>>",on_click=nextpage)
    
    tab = df_tr.iloc[[i+4,-1]].T

    tab.columns = ["match","moy"]
    left_column.title("Match n°"+str(i))
    left_column.table(tab.astype(str)) # .style.format("{:.2f}") # table # dataframe
    
    st.sidebar.table(df_pres)
    
    
elif st.session_state.cur_page == 5:
    
    st.title("Entraînement terminé.")
   
    df = st.session_state.reponses_num
    df.to_csv("results/reponses_matchs_tr_" + str(USER) + ".csv")
    with open("results/times_tr_" + str(USER) + ".pkl",'wb') as f:
        pickle.dump(st.session_state.click_list,f)
    
    "Vous allez passer à l'évaluation réelle de votre capacité de pronostic.\n"
    "Objectif : prédire l'équipe gagnante d'un match de League of Legends à partir de données statistiques à 10 minutes du match."
    "Vous aurez 55 matchs à évaluer : vous pouvez prendre une pause entre deux pronostics au besoin, mais une fois que l'écran de match s'affiche, tentez de répondre au plus vite."
    
    st.session_state.reponses_num = df.drop(df.index)
    
    st.session_state.cur_match = 0
    
    st.button("START>>>",key='nekst',on_click=nextpage)
    
    
elif st.session_state.cur_page == 6:
    
    left_column, right_column = st.columns(2)
    
    
    with right_column:
        
        i = st.session_state.cur_match
        st.title("Match n°"+str(i+1))
        
        title = 'Quel est votre pronostic ?'
        
        r1 = st.select_slider(title,options_ope,"je ne sais pas",key="slide")

        increment = st.button("NEXT>>>",on_click=nextexample)
    
    tab = df_ev.iloc[[i,-1]].T
    tab.columns = ["match","moy"]
    left_column.table(tab.astype(str)) 
    
    st.sidebar.table(df_pres)
    
elif st.session_state.cur_page ==7:
    
    left_column, right_column = st.columns(2)
    if st.session_state.cur_match<54:
        left_column.title("Paré ?")
        right_column.button("NEXT>>>",on_click=backtothree,args=[6])
    else:
        left_column.title("Fin des matchs !")
        "Il reste une dernière étape : remplir un formulaire rapide (moins de 5 minutes)."
        right_column.button("NEXT>>>",on_click=nextpage)
    
    st.sidebar.table(df_pres)


elif st.session_state.cur_page == 8:
    
    st.title("Questions sur votre expérience avec League of Legends")
    
    r1 = [0]*9
    
    l="À quelle fréquence jouez-vous à League of Legends ?"
    r1[0] = st.selectbox(l,temp_jeu)
    
    l="Quels rôles avez-vous déjà joué, jouez-vous usuellement ?"
    r1[1] = st.text_input(l)
    
    l="Avez-vous déjà joué en équipe avec des amis ? (et si oui, combien)"
    r1[2] = st.selectbox(l,["juste en solo",1,2,4])
    
    l="Depuis combien d'années jouez-vous à des MOBA ?"
    r1[3] = st.number_input(l, max_value=21,format="%d")
    
    l="Quel est votre rang actuel dans LoL ?"
    r1[4] = st.selectbox(l, rangs)
    
    l="À quelle fréquence regardez-vous des matchs compétitifs ?"
    r1[5] = st.selectbox(l,temp_visio)
    
    l="Depuis combien d'année vous êtes-vous intéressé à la méta de LoL ? (0 si vous l'ignorez)"
    r1[6] = st.number_input(l, max_value=14,format="%d")
    
    l="Avez-vous déjà publié du contenu grâce à LoL (vidéo, article) ? Gagnez-vous de l'argent graĉe à LoL ?"
    r1[7] = st.text_input(l)
    
    l="Avez-vous des connaissances en science des données ? (choisissez au plus proche)"
    r1[8] = st.select_slider(l,datascience)
    
    # st.write(r1)
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    

elif st.session_state.cur_page == 9:
    
    st.title("Ressentis au cours de l'expérience")
    
    r1 = st.select_slider("Pensez-vous avoir progressé au cours de l’utilisation ?",
                          nondutout,"indécis")
    
    r2 = st.select_slider("Étiez-vous très concentré durant l’expérience ?",
                          nondutout,"indécis")
    
    r3 = st.select_slider("L’expérience vous a-t-elle fatigué ?",
                          nondutout,"indécis")
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    
    
elif st.session_state.cur_page == 10:
    
    st.title("Informations générales")
    
    r1 = st.radio("Vous êtes : ", ["un homme","une femme","préfère ne pas répondre"])
    
    r2 = st.text_input("Quel âge avez-vous : ")
    
    r3 = st.text_input("Y a-t-il d'autres points à relever sur votre profil ou sur le déroulé de l'expérience ?")
    
    st.button("NEXT>>>",key='nekst',on_click=nextpage)
    
    
elif st.session_state.cur_page == 11:
    
    st.title("Félicitation, vous avez fini !")
    
    st.session_state.reponses_num.to_csv("results/reponses_matchs_ev_" 
                                          + str(USER) + ".csv")
    with open("results/times_ev_" + str(USER) + ".pkl",'wb') as f:
        pickle.dump(st.session_state.click_list,f)
    
    st.session_state.reponses_form.to_csv("results/reponses_formulaire_" 
                                          + str(USER) + ".csv")
    "Merci pour votre participation ! Vous pouvez arrêter le programme dans le terminal avec Ctrl+C, puis fermer l'onglet."
    
    
    
if st.session_state.t_print is None:
    st.session_state['t_print'] = time.time()
st.write("page : "+str(j))
clickable = True
