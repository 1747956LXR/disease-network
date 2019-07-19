

        return res

    # @profile
    def grad(self):
        grad_A = np.zeros_like(self.A)
        grad_u = np.zeros_like(self.u)

        # grad_A
        for d in range(self.D):
            for dk in range(self.D):

                gradient = 0
                for i in self.patients:
                    disease_d = (self.d[i] == d)
                    if disease_d.any():
                        tijs = self.t[i][disease_d]
                        tiks = self.t[i][self.d[i] == dk]

                        T = self.t[i][-1]

                        for tij in tijs:
                            intensity_ij = self.intensity(tij, i, d)
                            for tik in tiks:
                                if tik < tij:
                                    gradient += g(tij - tik) / intensity_ij

                                gradient -= G(T - tik)  #

                gradient = -gradient
                gradient += L1 * np.sign(grad_A[d][dk])
                grad_A[d][dk] = gradient

        # grad_u
        for d in range(self.D):
            gradient = 0
            for i in self.patients:
                disease_d = (self.d[i] == d)
                if disease_d.any():
                    gradient += sum(
                        self.f[i][disease_d][k] /
                        self.intensity(self.t[i][disease_d][k], i, d)
                        for k in range(np.sum(disease_d)))

                tijs = self.t[i]
                fijs = self.f[i]
                T = self.t[i][-1]

                gradient -= sum(fijs[j] * (tijs[j + 1] - tijs[j])
                                for j in range(len(tijs) - 1))
                # gradient -= fij * (tij - tij_1)

                # gradient -= fijs[-1] * (T - tijs[-1])

            gradient = -gradient
            gradient += L2 * self.u[d]
            grad_u = gradient
