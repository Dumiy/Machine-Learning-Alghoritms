""" definirea problemei"""


''' code for sudoku solving using A* '''
import copy


class Nod:
    NR_LINII = 9
    NR_COLOANE = 9
    NR_LINII_BLOC = 3
    NR_COLOANE_BLOC = 3

    def __init__(self, info, h = 0):
        self.info = info
        self.h = h

    def __str__(self):
        sir = "\n"
        for ind, i in enumerate(self.info):
            if ind % 3 == 0:
                sir += '\n'
            sir += '\n'
            for indj, j in enumerate(i):
                if indj % 3 == 0 and indj != 0:
                    sir += '   '
                if j == '0':
                    sir += "  "
                else:
                    sir += ' ' + str(j)


        sir += "   \n"
        return sir


    def __repr__(self):
        sir = '\n'
        sir += str(self.info)
        return sir


class Graf:
    def __init__(self, nod_start):
        self.nod_start = Nod(list(nod_start))
        # self.nod_scop = Nod(list(nod_scop), 0)

    def scop(self, nod):
        # verificati daca jocul s-a incheiat
        # ---------------------------------------
        # VERIFICA PE COLOANE
        # ---------------------------------------
        for i in range(Nod.NR_LINII):
            for j in range(Nod.NR_COLOANE):
                if nod.info[i][j] == 0:
                    return False
        return True

    def calculeaza_h(self, nod_info):
        # numarul de zero-uri din matrice, la care adaug numarul de linii, coloane, blocuri care nu sunt completate
        zerouri = 0
        for i in range(Nod.NR_LINII):
            for j in range(Nod.NR_COLOANE):
                if nod_info[i][j] == 0:
                    zerouri += 1
        # ---------------------------------------
        # VERIFICA PE COLOANE
        # ---------------------------------------
        nr_coloane = 0
        for j in range(Nod.NR_COLOANE):
            coloana = []
            for i in range(Nod.NR_LINII):
                coloana.append(nod_info[i][j])
            if 0 in coloana:
                nr_coloane += 1
        # ---------------------------------------
        # VERIFICA PE LINII
        # ---------------------------------------
        nr_linii = 0
        for linie in nod_info:
            if 0 in linie:
                nr_linii += 1
        # ---------------------------------------
        # VERIFICA IN CADRANE
        # ---------------------------------------
        nr_intervale = 0
        for i in range(0, Nod.NR_LINII, 3):
            for j in range(0, Nod.NR_COLOANE, 3):
                lista_interval = []
                for interval_linii in range(i, i + 3):
                    for interval_coloane in range(j, j + 3):
                        lista_interval.append(nod_info[interval_linii][interval_coloane])
                if 0 in lista_interval:
                    nr_intervale += 1
        return nr_linii+nr_coloane+nr_intervale

    def valid(self, matrice, i, j, element):
        lista = matrice[i]
        if element in lista:
            return False
        coloana = []
        for ii in range(Nod.NR_COLOANE):
            coloana.append(matrice[ii][j])
        if element in coloana:
            return False

        linie = int(i/3) * 3
        coloana = int(j/3) * 3
        cadran = []
        for ii in range(linie, linie + 3):
            for jj in range(coloana, coloana + 3):
                cadran.append(matrice[ii][jj])
        if element in cadran:
            return False
        return True

    def calculeaza_succesori(self, nod):
        l_succesori = []
        for i in range(Nod.NR_LINII):
            for j in range(Nod.NR_COLOANE):
                if nod.info[i][j] == 0:
                    for numar_nou in range(1,10):
                        if self.valid(nod.info, i, j, numar_nou):
                            matrice_copie = copy.deepcopy(nod.info)
                            matrice_copie[i][j] = numar_nou
                            h_nou = self.calculeaza_h(matrice_copie)
                            l_succesori.append((Nod(matrice_copie, h_nou),1))

        return l_succesori

""" Sfarsit definire problema """

""" Clase folosite in algoritmul A* """

class NodCautare:
    def __init__(self, nod_graf, succesori=[], parinte=None, g=0, f=None):
        self.nod_graf = nod_graf
        self.succesori = succesori
        self.parinte = parinte
        self.g = g
        if f is None:
            self.f = self.g + self.nod_graf.h
        else:
            self.f = f

    def drum_arbore(self):
        nod_c = self
        drum = [nod_c]
        while nod_c.parinte is not None:
            drum = [nod_c.parinte] + drum
            nod_c = nod_c.parinte
        return drum

    def contine_in_drum(self, nod):
        nod_c = self
        while nod_c.parinte is not None:
            if nod.info == nod_c.nod_graf.info:
                return True
            nod_c = nod_c.parinte
        return False

    def __str__(self):
        parinte = self.parinte if self.parinte is None else self.parinte.nod_graf.info
        # return "("+str(self.nod_graf)+", parinte="+", f="+str(self.f)+", g="+str(self.g)+")";
        return str(self.nod_graf)


""" Algoritmul A* """

def debug_str_l_noduri(l):
    sir = ""
    for x in l:
        sir += str(x) + "\n"
    return sir

def get_lista_solutii(l):
    drum = []
    for x in l:
         drum.append(x.nod_graf.info)
    return drum

def in_lista(l, nod):
    for x in l:
        if x.nod_graf.info == nod.info:
            return x
    return None

def a_star(graf):
    rad_arbore = NodCautare(nod_graf=graf.nod_start);
    open = [rad_arbore]
    closed = []
    drum_gasit = False
    while len(open) > 0:
        nod_curent = open.pop(0)
        closed.append(nod_curent)
        if graf.scop(nod_curent.nod_graf):
            drum_gasit = True
            break
        l_succesori = graf.calculeaza_succesori(nod_curent.nod_graf)
        for (nod, cost) in l_succesori:
            if (not nod_curent.contine_in_drum(nod)):
                x = in_lista(closed, nod)
                g_succesor = nod_curent.g + cost
                f = g_succesor + nod.h
                if x is not None:
                    if (f < nod_curent.f):
                        x.parinte = nod_curent
                        x.g = g_succesor
                        x.f = f
                else:
                    x = in_lista(open, nod)
                    if x is not None:
                        if (x.g > g_succesor):
                            x.parinte = nod_curent
                            x.g = g_succesor
                            x.f = f
                    else:  # cand nu e nici in closed nici in open
                        nod_cautare = NodCautare(nod_graf=nod, parinte=nod_curent,
                                                 g=g_succesor);  # se calculeaza f automat in constructor
                        open.append(nod_cautare)

        open.sort(key=lambda x: (x.f, -x.g))

    if drum_gasit == True:
        print("-----------------------------------------")
        print("Drum de cost minim: \n" + debug_str_l_noduri(nod_curent.drum_arbore()))
    else:
        print("\nNu exista solutie!")
        return []


    """ added by Cristina: """
    return get_lista_solutii(nod_curent.drum_arbore())



def main():

    nod_start = [
                    [7,9,0,    3,5,1,  6,4,0],
                    [6,4,0,    9,7,8,  0,3,0],
                    [3,0,0,    4,2,6,  7,9,0],

                    [9,0,4,    8,6,7,  5,2,3],
                    [2,0,0,    5,3,4,  8,1,9],
                    [8,5,3,    2,1,9,  4,7,6],

                    [1,0,0,    0,9,3,  2,5,4],
                    [5,3,0,    0,4,2,  9,8,7],
                    [4,2,9,    7,0,5,  3,6,1]
                ]
    # SCOPUL este acela de a inlocui cifrele 0 cu cifre intre 1 si 9, astfel incat aceeasi cifra sa nu apara de doua ori
    # pe aceeasi linie, coloana, sau in interiorul unui bloc de 3x3

    problema = Graf(nod_start)
    return a_star(problema)


# Apel:
main()