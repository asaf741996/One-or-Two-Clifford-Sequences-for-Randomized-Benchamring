import numpy as np
from scipy.linalg import sqrtm, inv
import pickle
from qutip import *


def swap(M):
    SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return np.dot(np.dot(SWAP, M), SWAP)


def Gates_multi(M1, M2):
    a = M1[0] + M2[0]
    b = np.dot(M1[1], M2[1])
    b = np.array(Qobj(b).tidyup())
    return [a, b]


def Is_New(M_array, M):
    for n in range(len(M_array)):
        a = np.absolute(np.trace(np.dot(M_array[n][1].conj().T, M[1])))
        b = int(np.sqrt(np.size(M[1])))
        if np.allclose(a, b):
            return False
        else:
            continue
    return True


class Clifford:
    def __init__(self, nQ):
        self.nQ = nQ
        self.Cliff = []
        if nQ == 1:
            self.element_number = 24
        else:
            self.element_number = 11520
        I = np.array([[1, 0], [0, 1]])
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        X_90 = sqrtm(X)
        Y_90 = sqrtm(Y)
        if nQ == 1:
            self.Prim = [[[0], I],
                         [[1], X],
                         [[2], Y],
                         [[3], X_90],
                         [[4], Y_90], ]
        else:
            ISWAP = np.array([[1, 0, 0, 0],
                              [0, 0, 1j, 0],
                              [0, 1j, 0, 0],
                              [0, 0, 0, 1]])
            self.Prim = [[[0], np.kron(I, I)],
                         [[1], np.kron(I, X)],
                         [[2], swap(np.kron(I, X))],
                         [[3], np.kron(I, Y)],
                         [[4], swap(np.kron(I, Y))],
                         [[5], np.kron(I, X_90)],
                         [[6], swap(np.kron(I, X_90))],
                         [[7], np.kron(I, Y_90)],
                         [[8], swap(np.kron(I, Y_90))],
                         [[9], ISWAP]]

    def generate_group(self):

        Cliff_index = []
        L = []
        c = 0
        count = 0
        # Iterate first Zv's
        for i in range(len(self.Prim)):
            a = self.Prim[i]
            if Is_New(self.Cliff, a):
                self.Cliff.append(a)
                Cliff_index.append(a[0])
                c += 1
                count += 1
            #         print(count)
            else:
                continue
        L.append(c)
        c = 0

        t = 1
        d = 0
        while True:

            if t > 1:
                d += L[t - 2]
            for i in range(d, len(self.Cliff)):
                # print('shi', self.Cliff_2[c][0])
                for j in range(len(self.Prim)):
                    a = Gates_multi(self.Prim[j], self.Cliff[i])
                    if Is_New(self.Cliff, a):
                        self.Cliff.append(a)
                        Cliff_index.append(a[0])
                        c += 1
                        count += 1
                        if count % 10 == 0:
                            print("cliff number", count)
                    if len(self.Cliff) == self.element_number:
                        break
                    else:
                        continue
                if len(self.Cliff) == self.element_number:
                    break
                else:
                    continue
            L.append(c)
            c = 0
            t += 1
            if len(self.Cliff) == self.element_number:
                break
            else:
                continue

    def save_group(self):
        if self.nQ == 1:
            with open('1q_Cliffords_gates.pkl', 'wb') as file:
                pickle.dump(self.Cliff, file)
        else:
            with open('2q_Cliffords_gates.pkl', 'wb') as file:
                pickle.dump(self.Cliff, file)

    def load_group(self):
        if self.nQ == 1:
            with open('1q_Cliffords_gates.pkl', 'rb') as f:
                self.Cliff = pickle.load(f)
        else:
            with open('2q_Cliffords_gates.pkl', 'rb') as f:
                self.Cliff = pickle.load(f)

    def find_inv_gate(self, matrix):
        # print(Qobj(matrix).tidyup())

        for n in range(len(self.Cliff)):
            if np.allclose(np.absolute(np.trace(np.dot(self.Cliff[n][1].conj().T, inv(matrix)))), 2 ** self.nQ):
                return self.Cliff[n][0]

    def random_clifford(self):
        i = np.random.randint(len(self.Cliff))
        return self.Cliff[i]

    def random_seq(self, seq_length, is_purity=False, interleaved_gates=None):
        circuit = []
        circuit_interleaved = []
        circuit_purity = []
        if is_purity is False:
            inv = self.Prim[0]
            inv_interleaved = self.Prim[0]
            for m in range(seq_length):
                g = self.random_clifford()
                circuit.append(g[0])
                inv = Gates_multi(inv, g)
                if interleaved_gates is not None:
                    circuit_interleaved.append(g[0])
                    circuit_interleaved.append(self.Prim[interleaved_gates][0])
                    inv_interleaved = Gates_multi(inv_interleaved, g)
                    inv_interleaved = Gates_multi(inv_interleaved, self.Prim[interleaved_gates])

            circuit.append(self.find_inv_gate(inv[1]))
            if interleaved_gates is None:
                return circuit
            else:
                circuit_interleaved.append(self.find_inv_gate(inv_interleaved[1]))
                return circuit, circuit_interleaved
        else:
            sub_circuit = []
            for m in range(seq_length):
                g = self.random_clifford()
                sub_circuit.append(g[0])
            for i in range(3):
                if self.nQ == 1:
                    circuit_purity.append(sub_circuit + [[i]])
                else:
                    for j in range(3):
                        circuit_purity.append(sub_circuit + [[i]] + [[j]])
            return circuit_purity

    def randomized_benchmarking_seq(self, nseeds=5, length_vector=None, is_purity=False, interleaved_gates=None):
        if length_vector is None:
            length_vector = [1, 25, 50, 75, 100]

        circuits = []
        circuits_interleaved = []
        circuits_purity = []

        if is_purity is False:
            for m in length_vector:
                circuits_seed = []
                circuits_interleaved_seed = []
                for seed in range(nseeds):
                    if interleaved_gates is None:
                        circuits_seed.append(self.random_seq(m))
                    elif interleaved_gates is not None:
                        a, b = self.random_seq(m, interleaved_gates=interleaved_gates)
                        circuits_seed.append(a)
                        circuits_interleaved_seed.append(b)
                circuits.append(circuits_seed)
                circuits_interleaved.append(circuits_interleaved_seed)
            if interleaved_gates is None:
                return circuits, length_vector
            else:
                return circuits, circuits_interleaved, length_vector

        else:
            for m in length_vector:
                circuits_purity_seed = []
                for seed in range(nseeds):
                    circuits_purity_seed.append(self.random_seq(m, is_purity=is_purity))
                circuits_purity.append(circuits_purity_seed)
            return circuits_purity, length_vector

    def check_rb_seq(self):

        for i in range(10):
            if self.nQ == 1:
                state = [1, 0]
            else:
                state = [1, 0, 0, 0]
            circ = self.random_seq(200)

            for cliff in circ:
                for gate in cliff:
                    state = self.Prim[gate][1] @ state

            print(abs(np.dot(state, [1, 0, 0, 0])))

    def get_unitary(self, keylist):
        M = np.identity(2 ** self.nQ)
        # print(self.Prim[0][1])
        for i in range(len(keylist)):
            M = self.Prim[keylist[i]][1] @ M
        return M

    def check(self):

        for i in range(len(self.Cliff) - 1):
            print(i, "is checking...")
            A = self.get_unitary(self.Cliff[i][0])
            for j in range(i + 1, len(self.Cliff)):
                B = self.get_unitary(self.Cliff[j][0])
                if np.allclose(np.absolute(np.trace(np.dot(A.conj().T, B))), 2 ** self.nQ):
                    print("error!", i, "and", j, "is the same!")
                    return 'Very bad'
        # if
        # print("success!")


C = Clifford(2)
C.load_group()
C.check_rb_seq()
