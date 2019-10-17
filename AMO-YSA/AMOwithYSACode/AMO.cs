using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AMOwithYSACode
{
    class AMO
    {
        int N = 100;//popsize
        int Max_iteration = 1000;
        int lb = -1; //
        int ub = 1; //
        int dim = 35;
        double[][] X;
        double[] fit;
        double[] GlobalParams;
        double GlobalMin = Double.MaxValue;

        int FES = 0;
        int maxFES = 30000;
        Random r = new Random();
        List<double> Convergence_curve = new List<double>();

        public AMO()
        {
            X = new double[N][];
            fit = new double[N];
            GlobalParams = new double[N];

            initialization();
        }

        public void initialization() //ilk değerlerin atandığı kısım
        {
            for (int i = 0; i < N; i++)
            {
                X[i] = new double[dim];

                for (int j = 0; j < dim; j++) // ilk değer atanıyor
                {
                    X[i][j] = lb + r.NextDouble() * (ub - lb);
                    //X[i][j] = -1 + r.NextDouble() * 2;
                }
                fit[i] = fonksiyon(X[i]); // fitness

                if (fit[i] < GlobalMin) // gbest bul
                {
                    GlobalMin = fit[i];
                    Array.Copy(X[i], GlobalParams, dim);// gbest doldur
                }
            }
        }


        public void run()
        {
            double[][] X_g1 = new double[N][];
            double[] fit_V = new double[N];
            double[] pa = new double[N];

            int iter = 0;
            while (FES <= maxFES)
            {
                iter++;
                update1(X, lb, ub);// sınır dışında ise, ilgili değeri sınır içinde tekrar üret

                for (int i = 0; i < N; i++)
                {
                    X_g1[i] = new double[dim];
                    double sigma = getNormalDistribution();

                    for (int j = 0; j < dim; j++)
                    {
                        int random_komsu = r.Next(0, 5);
                        int komsuluk = 0;
                        if (random_komsu == 0) komsuluk = -2;
                        if (random_komsu == 1) komsuluk = -1;
                        if (random_komsu == 2) komsuluk = 0;
                        if (random_komsu == 3) komsuluk = 1;
                        if (random_komsu == 4) komsuluk = 2;

                        int komsu = i + komsuluk;
                        komsu = MathMod(komsu, N);

                        X_g1[i][j] = X[i][j] + sigma * (X[komsu][j] - X[i][j]);
                    }// end of for dim
                }// end of for N

                update1(X_g1, lb, ub);// sınır dışında ise, ilgili değeri sınır içinde tekrar üret
                for (int i = 0; i < N; i++)
                {
                    fit_V[i] = fonksiyon(X_g1[i]);
                    if (fit_V[i] < fit[i])
                    {
                        fit[i] = fit_V[i];
                        Array.Copy(X_g1[i], X[i], dim);
                    }
                    if (fit[i] < GlobalMin) // gbest bul
                    {
                        GlobalMin = fit[i];
                        Array.Copy(X[i], GlobalParams, dim);// gbest doldur
                    }
                }

                int[] sortIndex = sort(fit);// sort

                for (int i = 0; i < N; i++)
                {
                    pa[sortIndex[i]] = 1 - ((double)i / N);// popülasyondaki en iyi bireyin pa değeri 1, bir sonraki 0.9, bir sonraki 0.8 ...
                }

                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < dim; j++)
                    {
                        int r1, r2;
                        while (true)
                        {
                            r1 = r.Next(0, N);
                            r2 = r.Next(0, N);
                            if (r1 != r2 && r1 != i && r2 != i)
                            {
                                break;
                            }
                        }

                        if (r.NextDouble() > pa[i])
                        {
                            X_g1[i][j] = X[r1][j] + r.NextDouble() * (GlobalParams[j] - X[i][j]) + r.NextDouble() * (X[r2][j] - X[i][j]);
                        }
                        else
                        {
                            X_g1[i][j] = X[i][j];
                        }
                    }// end of for dim
                }// end of for N

                update1(X_g1, lb, ub);// sınır dışında ise, ilgili değeri sınır içinde tekrar üret
                for (int i = 0; i < N; i++)
                {
                    fit_V[i] = fonksiyon(X_g1[i]);
                    if (fit_V[i] < fit[i])
                    {
                        fit[i] = fit_V[i];
                        Array.Copy(X_g1[i], X[i], dim);
                    }
                    if (fit[i] < GlobalMin) // gbest bul
                    {
                        GlobalMin = fit[i];
                        Array.Copy(X[i], GlobalParams, dim);// gbest doldur
                    }
                }

                Convergence_curve.Add(GlobalMin);

            }

            Ysa_Test(GlobalParams);
        }

        private static int[] sort(double[] dizi) // diziyi sıralar ve sıralanmış indexini dönderir
        {
            List<double> A = dizi.ToList();

            var sorted = A
                .Select((x, i) => new KeyValuePair<double, int>(x, i))
                .OrderBy(x => x.Key)
                .ToList();

            List<double> B = sorted.Select(x => x.Key).ToList();
            List<int> idx = sorted.Select(x => x.Value).ToList();
            return idx.ToArray();
        }

        double getNormalDistribution() // gaussian dağılımına uygun random sayı üretiyor
        {
            double u1 = 1.0 - r.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - r.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return randStdNormal;
        }

        int MathMod(int a, int b)// mod alıyor. MathMod(-2, 30) --> 28 , MathMod(-1, 30) --> 29 , MathMod(1, 30) --> 1
        {
            int res = (Math.Abs(a * b) + a) % b;
            return res;
        }

        private void update1(double[][] p, int lb, int ub)// sınır dışında ise, ilgili değeri sınır içinde tekrar üret
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    if (p[i][j] < lb || p[i][j] > ub)
                        p[i][j] = lb + r.NextDouble() * (ub - lb);
                }
            }
        }

        private double fonksiyon(double[] cozum)
        {
            FES++;
            return Ysa_Train(cozum);
        }

        private double Ysa_Train(double[] AnimalValues)
        {
            double[][] DataSet = new double[150][];
            //İRİS-SETOSA
            DataSet[0] = new double[] { 5.1, 3.5, 1.4, 0.2, 0, 0, 1 };
            DataSet[1] = new double[] { 4.9, 3.0, 1.4, 0.2, 0, 0, 1 };
            DataSet[2] = new double[] { 4.7, 3.2, 1.3, 0.2, 0, 0, 1 };
            DataSet[3] = new double[] { 4.6, 3.1, 1.5, 0.2, 0, 0, 1 };
            DataSet[4] = new double[] { 5.0, 3.6, 1.4, 0.2, 0, 0, 1 };
            DataSet[5] = new double[] { 5.4, 3.9, 1.7, 0.4, 0, 0, 1 };
            DataSet[6] = new double[] { 4.6, 3.4, 1.4, 0.3, 0, 0, 1 };
            DataSet[7] = new double[] { 5.0, 3.4, 1.5, 0.2, 0, 0, 1 };
            DataSet[8] = new double[] { 4.4, 2.9, 1.4, 0.2, 0, 0, 1 };
            DataSet[9] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
            DataSet[10] = new double[] { 5.4, 3.7, 1.5, 0.2, 0, 0, 1 };
            DataSet[11] = new double[] { 4.8, 3.4, 1.6, 0.2, 0, 0, 1 };
            DataSet[12] = new double[] { 4.8, 3.0, 1.4, 0.1, 0, 0, 1 };
            DataSet[13] = new double[] { 4.3, 3.0, 1.1, 0.1, 0, 0, 1 };
            DataSet[14] = new double[] { 5.8, 4.0, 1.2, 0.2, 0, 0, 1 };
            DataSet[15] = new double[] { 5.7, 4.4, 1.5, 0.4, 0, 0, 1 };
            DataSet[16] = new double[] { 5.4, 3.9, 1.3, 0.4, 0, 0, 1 };
            DataSet[17] = new double[] { 5.1, 3.5, 1.4, 0.3, 0, 0, 1 };
            DataSet[18] = new double[] { 5.7, 3.8, 1.7, 0.3, 0, 0, 1 };
            DataSet[19] = new double[] { 5.1, 3.8, 1.5, 0.3, 0, 0, 1 };
            DataSet[20] = new double[] { 5.4, 3.4, 1.7, 0.2, 0, 0, 1 };
            DataSet[21] = new double[] { 5.1, 3.7, 1.5, 0.4, 0, 0, 1 };
            DataSet[22] = new double[] { 4.6, 3.6, 1.0, 0.2, 0, 0, 1 };
            DataSet[23] = new double[] { 5.1, 3.3, 1.7, 0.5, 0, 0, 1 };
            DataSet[24] = new double[] { 4.8, 3.4, 1.9, 0.2, 0, 0, 1 };
            DataSet[25] = new double[] { 5.0, 3.0, 1.6, 0.2, 0, 0, 1 };
            DataSet[26] = new double[] { 5.0, 3.4, 1.6, 0.4, 0, 0, 1 };
            DataSet[27] = new double[] { 5.2, 3.5, 1.5, 0.2, 0, 0, 1 };
            DataSet[28] = new double[] { 5.2, 3.4, 1.4, 0.2, 0, 0, 1 };
            DataSet[29] = new double[] { 4.7, 3.2, 1.6, 0.2, 0, 0, 1 };
            DataSet[30] = new double[] { 4.8, 3.1, 1.6, 0.2, 0, 0, 1 };
            DataSet[31] = new double[] { 5.4, 3.4, 1.5, 0.4, 0, 0, 1 };
            DataSet[32] = new double[] { 5.2, 4.1, 1.5, 0.1, 0, 0, 1 };
            DataSet[33] = new double[] { 5.5, 4.2, 1.4, 0.2, 0, 0, 1 };
            DataSet[34] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
            DataSet[35] = new double[] { 5.0, 3.2, 1.2, 0.2, 0, 0, 1 };
            DataSet[36] = new double[] { 5.5, 3.5, 1.3, 0.2, 0, 0, 1 };
            DataSet[37] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
            DataSet[38] = new double[] { 4.4, 3.0, 1.3, 0.2, 0, 0, 1 };
            DataSet[39] = new double[] { 5.1, 3.4, 1.5, 0.2, 0, 0, 1 };
            DataSet[40] = new double[] { 5.0, 3.5, 1.3, 0.3, 0, 0, 1 };
            DataSet[41] = new double[] { 4.5, 2.3, 1.3, 0.3, 0, 0, 1 };
            DataSet[42] = new double[] { 4.4, 3.2, 1.3, 0.2, 0, 0, 1 };
            DataSet[43] = new double[] { 5.0, 3.5, 1.6, 0.6, 0, 0, 1 };
            DataSet[44] = new double[] { 5.1, 3.8, 1.9, 0.4, 0, 0, 1 };
            DataSet[45] = new double[] { 4.8, 3.0, 1.4, 0.3, 0, 0, 1 };
            DataSet[46] = new double[] { 5.1, 3.8, 1.6, 0.2, 0, 0, 1 };
            DataSet[47] = new double[] { 4.6, 3.2, 1.4, 0.2, 0, 0, 1 };
            DataSet[48] = new double[] { 5.3, 3.7, 1.5, 0.2, 0, 0, 1 };
            DataSet[49] = new double[] { 5.0, 3.3, 1.4, 0.2, 0, 0, 1 };
            //İRİS-VERSİCOLOR
            DataSet[50] = new double[] { 7.0, 3.2, 4.7, 1.4, 0, 1, 0 };
            DataSet[51] = new double[] { 6.4, 3.2, 4.5, 1.5, 0, 1, 0 };
            DataSet[52] = new double[] { 6.9, 3.1, 4.9, 1.5, 0, 1, 0 };
            DataSet[53] = new double[] { 5.5, 2.3, 4.0, 1.3, 0, 1, 0 };
            DataSet[54] = new double[] { 6.5, 2.8, 4.6, 1.5, 0, 1, 0 };
            DataSet[55] = new double[] { 5.7, 2.8, 4.5, 1.3, 0, 1, 0 };
            DataSet[56] = new double[] { 6.3, 3.3, 4.7, 1.6, 0, 1, 0 };
            DataSet[57] = new double[] { 4.9, 2.4, 3.3, 1.0, 0, 1, 0 };
            DataSet[58] = new double[] { 6.6, 2.9, 4.6, 1.3, 0, 1, 0 };
            DataSet[59] = new double[] { 5.2, 2.7, 3.9, 1.4, 0, 1, 0 };
            DataSet[60] = new double[] { 5.0, 2.0, 3.5, 1.0, 0, 1, 0 };
            DataSet[61] = new double[] { 5.9, 3.0, 4.2, 1.5, 0, 1, 0 };
            DataSet[62] = new double[] { 6.0, 2.2, 4.0, 1.0, 0, 1, 0 };
            DataSet[63] = new double[] { 6.1, 2.9, 4.7, 1.4, 0, 1, 0 };
            DataSet[64] = new double[] { 5.6, 2.9, 3.6, 1.3, 0, 1, 0 };
            DataSet[65] = new double[] { 6.7, 3.1, 4.4, 1.4, 0, 1, 0 };
            DataSet[66] = new double[] { 5.6, 3.0, 4.5, 1.5, 0, 1, 0 };
            DataSet[67] = new double[] { 5.8, 2.7, 4.1, 1.0, 0, 1, 0 };
            DataSet[68] = new double[] { 6.2, 2.2, 4.5, 1.5, 0, 1, 0 };
            DataSet[69] = new double[] { 5.6, 2.5, 3.9, 1.1, 0, 1, 0 };
            DataSet[70] = new double[] { 5.9, 3.2, 4.8, 1.8, 0, 1, 0 };
            DataSet[71] = new double[] { 6.1, 2.8, 4.0, 1.3, 0, 1, 0 };
            DataSet[72] = new double[] { 6.3, 2.5, 4.9, 1.5, 0, 1, 0 };
            DataSet[73] = new double[] { 6.1, 2.8, 4.7, 1.2, 0, 1, 0 };
            DataSet[74] = new double[] { 6.4, 2.9, 4.3, 1.3, 0, 1, 0 };
            DataSet[75] = new double[] { 6.6, 3.0, 4.4, 1.4, 0, 1, 0 };
            DataSet[76] = new double[] { 6.8, 2.8, 4.8, 1.4, 0, 1, 0 };
            DataSet[77] = new double[] { 6.7, 3.0, 5.0, 1.7, 0, 1, 0 };
            DataSet[78] = new double[] { 6.0, 2.9, 4.5, 1.5, 0, 1, 0 };
            DataSet[79] = new double[] { 5.7, 2.6, 3.5, 1.0, 0, 1, 0 };
            DataSet[80] = new double[] { 5.5, 2.4, 3.8, 1.1, 0, 1, 0 };
            DataSet[81] = new double[] { 5.5, 2.4, 3.7, 1.0, 0, 1, 0 };
            DataSet[82] = new double[] { 5.8, 2.7, 3.9, 1.2, 0, 1, 0 };
            DataSet[83] = new double[] { 6.0, 2.7, 5.1, 1.6, 0, 1, 0 };
            DataSet[84] = new double[] { 5.4, 3.0, 4.5, 1.5, 0, 1, 0 };
            DataSet[85] = new double[] { 6.0, 3.4, 4.5, 1.6, 0, 1, 0 };
            DataSet[86] = new double[] { 6.7, 3.1, 4.7, 1.5, 0, 1, 0 };
            DataSet[87] = new double[] { 6.3, 2.3, 4.4, 1.3, 0, 1, 0 };
            DataSet[88] = new double[] { 5.6, 3.0, 4.1, 1.3, 0, 1, 0 };
            DataSet[89] = new double[] { 5.5, 2.5, 4.0, 1.3, 0, 1, 0 };
            DataSet[90] = new double[] { 5.5, 2.6, 4.4, 1.2, 0, 1, 0 };
            DataSet[91] = new double[] { 6.1, 3.0, 4.6, 1.4, 0, 1, 0 };
            DataSet[92] = new double[] { 5.8, 2.6, 4.0, 1.2, 0, 1, 0 };
            DataSet[93] = new double[] { 5.0, 2.3, 3.3, 1.0, 0, 1, 0 };
            DataSet[94] = new double[] { 5.6, 2.7, 4.2, 1.3, 0, 1, 0 };
            DataSet[95] = new double[] { 5.7, 3.0, 4.2, 1.2, 0, 1, 0 };
            DataSet[96] = new double[] { 5.7, 2.9, 4.2, 1.3, 0, 1, 0 };
            DataSet[97] = new double[] { 6.2, 2.9, 4.3, 1.3, 0, 1, 0 };
            DataSet[98] = new double[] { 5.1, 2.5, 3.0, 1.1, 0, 1, 0 };
            DataSet[99] = new double[] { 5.7, 2.8, 4.1, 1.3, 0, 1, 0 };
            //İRİS-VİRGİNİCA
            DataSet[100] = new double[] { 6.3, 3.3, 6.0, 2.5, 1, 0, 0 };
            DataSet[101] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
            DataSet[102] = new double[] { 7.1, 3.0, 5.9, 2.1, 1, 0, 0 };
            DataSet[103] = new double[] { 6.3, 2.9, 5.6, 1.8, 1, 0, 0 };
            DataSet[104] = new double[] { 6.5, 3.0, 5.8, 2.2, 1, 0, 0 };
            DataSet[105] = new double[] { 7.6, 3.0, 6.6, 2.1, 1, 0, 0 };
            DataSet[106] = new double[] { 4.9, 2.5, 4.5, 1.7, 1, 0, 0 };
            DataSet[107] = new double[] { 7.3, 2.9, 6.3, 1.8, 1, 0, 0 };
            DataSet[108] = new double[] { 6.7, 2.5, 5.8, 1.8, 1, 0, 0 };
            DataSet[109] = new double[] { 7.2, 3.6, 6.1, 2.5, 1, 0, 0 };
            DataSet[110] = new double[] { 6.5, 3.2, 5.1, 2.0, 1, 0, 0 };
            DataSet[111] = new double[] { 6.4, 2.7, 5.3, 1.9, 1, 0, 0 };
            DataSet[112] = new double[] { 6.8, 3.0, 5.5, 2.1, 1, 0, 0 };
            DataSet[113] = new double[] { 5.7, 2.5, 5.0, 2.0, 1, 0, 0 };
            DataSet[114] = new double[] { 5.8, 2.8, 5.1, 2.4, 1, 0, 0 };
            DataSet[115] = new double[] { 6.4, 3.2, 5.3, 2.3, 1, 0, 0 };
            DataSet[116] = new double[] { 6.5, 3.0, 5.5, 1.8, 1, 0, 0 };
            DataSet[117] = new double[] { 7.7, 3.8, 6.7, 2.2, 1, 0, 0 };
            DataSet[118] = new double[] { 7.7, 2.6, 6.9, 2.3, 1, 0, 0 };
            DataSet[119] = new double[] { 6.0, 2.2, 5.0, 1.5, 1, 0, 0 };
            DataSet[120] = new double[] { 6.9, 3.2, 5.7, 2.3, 1, 0, 0 };
            DataSet[121] = new double[] { 5.6, 2.8, 4.9, 2.0, 1, 0, 0 };
            DataSet[122] = new double[] { 7.7, 2.8, 6.7, 2.0, 1, 0, 0 };
            DataSet[123] = new double[] { 6.3, 2.7, 4.9, 1.8, 1, 0, 0 };
            DataSet[124] = new double[] { 6.7, 3.3, 5.7, 2.1, 1, 0, 0 };
            DataSet[125] = new double[] { 7.2, 3.2, 6.0, 1.8, 1, 0, 0 };
            DataSet[126] = new double[] { 6.2, 2.8, 4.8, 1.8, 1, 0, 0 };
            DataSet[127] = new double[] { 6.1, 3.0, 4.9, 1.8, 1, 0, 0 };
            DataSet[128] = new double[] { 6.4, 2.8, 5.6, 2.1, 1, 0, 0 };
            DataSet[129] = new double[] { 7.2, 3.0, 5.8, 1.6, 1, 0, 0 };
            DataSet[130] = new double[] { 7.4, 2.8, 6.1, 1.9, 1, 0, 0 };
            DataSet[131] = new double[] { 7.9, 3.8, 6.4, 2.0, 1, 0, 0 };
            DataSet[132] = new double[] { 6.4, 2.8, 5.6, 2.2, 1, 0, 0 };
            DataSet[133] = new double[] { 6.3, 2.8, 5.1, 1.5, 1, 0, 0 };
            DataSet[134] = new double[] { 6.1, 2.6, 5.6, 1.4, 1, 0, 0 };
            DataSet[135] = new double[] { 7.7, 3.0, 6.1, 2.3, 1, 0, 0 };
            DataSet[136] = new double[] { 6.3, 3.4, 5.6, 2.4, 1, 0, 0 };
            DataSet[137] = new double[] { 6.4, 3.1, 5.5, 1.8, 1, 0, 0 };
            DataSet[138] = new double[] { 6.0, 3.0, 4.8, 1.8, 1, 0, 0 };
            DataSet[139] = new double[] { 6.9, 3.1, 5.4, 2.1, 1, 0, 0 };
            DataSet[140] = new double[] { 6.7, 3.1, 5.6, 2.4, 1, 0, 0 };
            DataSet[141] = new double[] { 6.9, 3.1, 5.1, 2.3, 1, 0, 0 };
            DataSet[142] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
            DataSet[143] = new double[] { 6.8, 3.2, 5.9, 2.3, 1, 0, 0 };
            DataSet[144] = new double[] { 6.7, 3.3, 5.7, 2.5, 1, 0, 0 };
            DataSet[145] = new double[] { 6.7, 3.0, 5.2, 2.3, 1, 0, 0 };
            DataSet[146] = new double[] { 6.3, 2.5, 5.0, 1.9, 1, 0, 0 };
            DataSet[147] = new double[] { 6.5, 3.0, 5.2, 2.0, 1, 0, 0 };
            DataSet[148] = new double[] { 6.2, 3.4, 5.4, 2.3, 1, 0, 0 };
            DataSet[149] = new double[] { 5.9, 3.0, 5.1, 1.8, 1, 0, 0 };

            int trainDataCount = (DataSet.Length * 70) / 100;
            int testDataCount = (DataSet.Length * 30) / 100;

            Random r = new Random();

            int[] trainDataNumbers = new int[trainDataCount];
            for (int i = 0; i < trainDataNumbers.Length; i++)
            {
                trainDataNumbers[i] = r.Next(0, trainDataCount);
            }

            int[] testDataNumbers = new int[testDataCount];
            for (int i = 0; i < testDataNumbers.Length; i++)
            {
                testDataNumbers[i] = r.Next(0, testDataCount);
            }
            /*
            for (int i = 0; i < trainDataNumbers.Length; i++)
            {
                Console.WriteLine(i+"="+trainDataNumbers[i]);
            }

            for (int i = 0; i < testDataNumbers.Length; i++)
            {
                Console.WriteLine(i + "=" + testDataNumbers[i]); 
            }
            */
            double[][] trainData = new double[trainDataCount][];
            for (int j = 0; j < trainData.Length; j++)
            {
                trainData[j] = DataSet[trainDataNumbers[j]];
            }
            /*
            for(int a=0;a<trainData.Length;a++)
            {
                for(int b=0;b<DataSet[0].Length;b++)
                {
                    Console.Write("[]" + a + "[]" + b + "=" + trainData[a][b]+"\n");
                }
                Console.WriteLine();
            }*/

            double[][] testData = new double[testDataCount][];
            for (int j = 0; j < testData.Length; j++)
            {
                testData[j] = DataSet[testDataNumbers[j]];
            }
            /*
            for (int a = 0; a < testData.Length; a++)
            {
                for (int b = 0; b < DataSet[0].Length; b++)
                {
                    Console.Write("[]" + a + "[]" + b + "=" + testData[a][b] + "\n");
                }
                Console.WriteLine();
            }*/

            int iNeuronCount = 4;
            int hNeuronCount = 4;
            int oNeuronCount = 3;

            double[] input = new double[4];
            double[] output = new double[3];
            double[] target = new double[3];
            double[] hidden = new double[4];

            double[][] ihWeihgts = new double[4][];
            double[] hBiases = new double[4];
            double[][] hoWeihgts = new double[4][];
            double[] oBiases = new double[3];

            int tmp = 0;
            for (int i = 0; i < iNeuronCount; i++)
            {
                ihWeihgts[i] = new double[4];
                for (int j = 0; j < hNeuronCount; j++)
                {
                    ihWeihgts[i][j] = AnimalValues[tmp];
                    tmp++;
                }
            }//ih agırlıklar
            for (int i = 0; i < hNeuronCount; i++)
            {
                hBiases[i] = AnimalValues[tmp];
                tmp++;
            }//hidden biası

            for (int i = 0; i < hNeuronCount; i++)
            {
                hoWeihgts[i] = new double[4];
                for (int j = 0; j < oNeuronCount; j++)
                {
                    hoWeihgts[i][j] = AnimalValues[tmp];
                    tmp++;
                }
            }

            for (int i = 0; i < oNeuronCount; i++)
            {
                oBiases[i] = AnimalValues[tmp];
                tmp++;
            }

            double[][] targets = new double[trainData.Length][];
            double[][] outputs = new double[trainData.Length][];

            //MSE Hesaplanarak Fit Değerleri bulunacak
            for (int i = 0; i < trainData.Length; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    input[j] = trainData[i][j];
                }
                int sayac = 0;
                for (int m = 4; m < trainData[i].Length; m++)
                {
                    target[sayac++] = trainData[i][m];
                }

                double[] hSum = new double[hNeuronCount];

                for (int j = 0; j < hNeuronCount; j++)
                {
                    for (int k = 0; k < iNeuronCount; k++)
                    {
                        hSum[j] += input[k] * ihWeihgts[k][j]; //ara katmandaki nöronların değerlerini hesapladık.
                    }
                    hSum[j] += hBiases[j];
                    hSum[j] = Sigmoid(hSum[j]);
                }

                double[] oSum = new double[oNeuronCount];

                for (int j = 0; j < oNeuronCount; j++)
                {
                    for (int k = 0; k < hNeuronCount; k++)
                    {
                        oSum[j] += hSum[k] * hoWeihgts[k][j]; //çıkış katmanındaki nöronlarımızın değerlerini hesapladık.
                    }
                    oSum[j] += oBiases[j];
                    output[j] = Sigmoid(oSum[j]);
                }

                
                targets[i] = new double[trainData.Length];
                for (int j = 0; j < target.Length; j++)
                {
                    targets[i][j] = target[j];
                }

                
                outputs[i] = new double[trainData.Length];
                for (int j = 0; j < output.Length; j++)
                {
                    outputs[i][j] = output[j];
                }
            }

            double MSE = 0;
            double M = 0;
            for (int u = 0; u < targets.Length; u++)
            {
                for (int w = 0; w < targets[0].Length; w++)
                {
                    MSE += Math.Pow((targets[u][w] - outputs[u][w]), 2);
                    M += targets[u][w];
                }
            }
            double pay = MSE;
            MSE /= targets.Length;

            return MSE;
        }

        private void Ysa_Test(double[] BestAnimalValues)
        {
            double[][] DataSet = new double[150][];
            //İRİS-SETOSA
            DataSet[0] = new double[] { 5.1, 3.5, 1.4, 0.2, 0, 0, 1 };
            DataSet[1] = new double[] { 4.9, 3.0, 1.4, 0.2, 0, 0, 1 };
            DataSet[2] = new double[] { 4.7, 3.2, 1.3, 0.2, 0, 0, 1 };
            DataSet[3] = new double[] { 4.6, 3.1, 1.5, 0.2, 0, 0, 1 };
            DataSet[4] = new double[] { 5.0, 3.6, 1.4, 0.2, 0, 0, 1 };
            DataSet[5] = new double[] { 5.4, 3.9, 1.7, 0.4, 0, 0, 1 };
            DataSet[6] = new double[] { 4.6, 3.4, 1.4, 0.3, 0, 0, 1 };
            DataSet[7] = new double[] { 5.0, 3.4, 1.5, 0.2, 0, 0, 1 };
            DataSet[8] = new double[] { 4.4, 2.9, 1.4, 0.2, 0, 0, 1 };
            DataSet[9] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
            DataSet[10] = new double[] { 5.4, 3.7, 1.5, 0.2, 0, 0, 1 };
            DataSet[11] = new double[] { 4.8, 3.4, 1.6, 0.2, 0, 0, 1 };
            DataSet[12] = new double[] { 4.8, 3.0, 1.4, 0.1, 0, 0, 1 };
            DataSet[13] = new double[] { 4.3, 3.0, 1.1, 0.1, 0, 0, 1 };
            DataSet[14] = new double[] { 5.8, 4.0, 1.2, 0.2, 0, 0, 1 };
            DataSet[15] = new double[] { 5.7, 4.4, 1.5, 0.4, 0, 0, 1 };
            DataSet[16] = new double[] { 5.4, 3.9, 1.3, 0.4, 0, 0, 1 };
            DataSet[17] = new double[] { 5.1, 3.5, 1.4, 0.3, 0, 0, 1 };
            DataSet[18] = new double[] { 5.7, 3.8, 1.7, 0.3, 0, 0, 1 };
            DataSet[19] = new double[] { 5.1, 3.8, 1.5, 0.3, 0, 0, 1 };
            DataSet[20] = new double[] { 5.4, 3.4, 1.7, 0.2, 0, 0, 1 };
            DataSet[21] = new double[] { 5.1, 3.7, 1.5, 0.4, 0, 0, 1 };
            DataSet[22] = new double[] { 4.6, 3.6, 1.0, 0.2, 0, 0, 1 };
            DataSet[23] = new double[] { 5.1, 3.3, 1.7, 0.5, 0, 0, 1 };
            DataSet[24] = new double[] { 4.8, 3.4, 1.9, 0.2, 0, 0, 1 };
            DataSet[25] = new double[] { 5.0, 3.0, 1.6, 0.2, 0, 0, 1 };
            DataSet[26] = new double[] { 5.0, 3.4, 1.6, 0.4, 0, 0, 1 };
            DataSet[27] = new double[] { 5.2, 3.5, 1.5, 0.2, 0, 0, 1 };
            DataSet[28] = new double[] { 5.2, 3.4, 1.4, 0.2, 0, 0, 1 };
            DataSet[29] = new double[] { 4.7, 3.2, 1.6, 0.2, 0, 0, 1 };
            DataSet[30] = new double[] { 4.8, 3.1, 1.6, 0.2, 0, 0, 1 };
            DataSet[31] = new double[] { 5.4, 3.4, 1.5, 0.4, 0, 0, 1 };
            DataSet[32] = new double[] { 5.2, 4.1, 1.5, 0.1, 0, 0, 1 };
            DataSet[33] = new double[] { 5.5, 4.2, 1.4, 0.2, 0, 0, 1 };
            DataSet[34] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
            DataSet[35] = new double[] { 5.0, 3.2, 1.2, 0.2, 0, 0, 1 };
            DataSet[36] = new double[] { 5.5, 3.5, 1.3, 0.2, 0, 0, 1 };
            DataSet[37] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };
            DataSet[38] = new double[] { 4.4, 3.0, 1.3, 0.2, 0, 0, 1 };
            DataSet[39] = new double[] { 5.1, 3.4, 1.5, 0.2, 0, 0, 1 };
            DataSet[40] = new double[] { 5.0, 3.5, 1.3, 0.3, 0, 0, 1 };
            DataSet[41] = new double[] { 4.5, 2.3, 1.3, 0.3, 0, 0, 1 };
            DataSet[42] = new double[] { 4.4, 3.2, 1.3, 0.2, 0, 0, 1 };
            DataSet[43] = new double[] { 5.0, 3.5, 1.6, 0.6, 0, 0, 1 };
            DataSet[44] = new double[] { 5.1, 3.8, 1.9, 0.4, 0, 0, 1 };
            DataSet[45] = new double[] { 4.8, 3.0, 1.4, 0.3, 0, 0, 1 };
            DataSet[46] = new double[] { 5.1, 3.8, 1.6, 0.2, 0, 0, 1 };
            DataSet[47] = new double[] { 4.6, 3.2, 1.4, 0.2, 0, 0, 1 };
            DataSet[48] = new double[] { 5.3, 3.7, 1.5, 0.2, 0, 0, 1 };
            DataSet[49] = new double[] { 5.0, 3.3, 1.4, 0.2, 0, 0, 1 };
            //İRİS-VERSİCOLOR
            DataSet[50] = new double[] { 7.0, 3.2, 4.7, 1.4, 0, 1, 0 };
            DataSet[51] = new double[] { 6.4, 3.2, 4.5, 1.5, 0, 1, 0 };
            DataSet[52] = new double[] { 6.9, 3.1, 4.9, 1.5, 0, 1, 0 };
            DataSet[53] = new double[] { 5.5, 2.3, 4.0, 1.3, 0, 1, 0 };
            DataSet[54] = new double[] { 6.5, 2.8, 4.6, 1.5, 0, 1, 0 };
            DataSet[55] = new double[] { 5.7, 2.8, 4.5, 1.3, 0, 1, 0 };
            DataSet[56] = new double[] { 6.3, 3.3, 4.7, 1.6, 0, 1, 0 };
            DataSet[57] = new double[] { 4.9, 2.4, 3.3, 1.0, 0, 1, 0 };
            DataSet[58] = new double[] { 6.6, 2.9, 4.6, 1.3, 0, 1, 0 };
            DataSet[59] = new double[] { 5.2, 2.7, 3.9, 1.4, 0, 1, 0 };
            DataSet[60] = new double[] { 5.0, 2.0, 3.5, 1.0, 0, 1, 0 };
            DataSet[61] = new double[] { 5.9, 3.0, 4.2, 1.5, 0, 1, 0 };
            DataSet[62] = new double[] { 6.0, 2.2, 4.0, 1.0, 0, 1, 0 };
            DataSet[63] = new double[] { 6.1, 2.9, 4.7, 1.4, 0, 1, 0 };
            DataSet[64] = new double[] { 5.6, 2.9, 3.6, 1.3, 0, 1, 0 };
            DataSet[65] = new double[] { 6.7, 3.1, 4.4, 1.4, 0, 1, 0 };
            DataSet[66] = new double[] { 5.6, 3.0, 4.5, 1.5, 0, 1, 0 };
            DataSet[67] = new double[] { 5.8, 2.7, 4.1, 1.0, 0, 1, 0 };
            DataSet[68] = new double[] { 6.2, 2.2, 4.5, 1.5, 0, 1, 0 };
            DataSet[69] = new double[] { 5.6, 2.5, 3.9, 1.1, 0, 1, 0 };
            DataSet[70] = new double[] { 5.9, 3.2, 4.8, 1.8, 0, 1, 0 };
            DataSet[71] = new double[] { 6.1, 2.8, 4.0, 1.3, 0, 1, 0 };
            DataSet[72] = new double[] { 6.3, 2.5, 4.9, 1.5, 0, 1, 0 };
            DataSet[73] = new double[] { 6.1, 2.8, 4.7, 1.2, 0, 1, 0 };
            DataSet[74] = new double[] { 6.4, 2.9, 4.3, 1.3, 0, 1, 0 };
            DataSet[75] = new double[] { 6.6, 3.0, 4.4, 1.4, 0, 1, 0 };
            DataSet[76] = new double[] { 6.8, 2.8, 4.8, 1.4, 0, 1, 0 };
            DataSet[77] = new double[] { 6.7, 3.0, 5.0, 1.7, 0, 1, 0 };
            DataSet[78] = new double[] { 6.0, 2.9, 4.5, 1.5, 0, 1, 0 };
            DataSet[79] = new double[] { 5.7, 2.6, 3.5, 1.0, 0, 1, 0 };
            DataSet[80] = new double[] { 5.5, 2.4, 3.8, 1.1, 0, 1, 0 };
            DataSet[81] = new double[] { 5.5, 2.4, 3.7, 1.0, 0, 1, 0 };
            DataSet[82] = new double[] { 5.8, 2.7, 3.9, 1.2, 0, 1, 0 };
            DataSet[83] = new double[] { 6.0, 2.7, 5.1, 1.6, 0, 1, 0 };
            DataSet[84] = new double[] { 5.4, 3.0, 4.5, 1.5, 0, 1, 0 };
            DataSet[85] = new double[] { 6.0, 3.4, 4.5, 1.6, 0, 1, 0 };
            DataSet[86] = new double[] { 6.7, 3.1, 4.7, 1.5, 0, 1, 0 };
            DataSet[87] = new double[] { 6.3, 2.3, 4.4, 1.3, 0, 1, 0 };
            DataSet[88] = new double[] { 5.6, 3.0, 4.1, 1.3, 0, 1, 0 };
            DataSet[89] = new double[] { 5.5, 2.5, 4.0, 1.3, 0, 1, 0 };
            DataSet[90] = new double[] { 5.5, 2.6, 4.4, 1.2, 0, 1, 0 };
            DataSet[91] = new double[] { 6.1, 3.0, 4.6, 1.4, 0, 1, 0 };
            DataSet[92] = new double[] { 5.8, 2.6, 4.0, 1.2, 0, 1, 0 };
            DataSet[93] = new double[] { 5.0, 2.3, 3.3, 1.0, 0, 1, 0 };
            DataSet[94] = new double[] { 5.6, 2.7, 4.2, 1.3, 0, 1, 0 };
            DataSet[95] = new double[] { 5.7, 3.0, 4.2, 1.2, 0, 1, 0 };
            DataSet[96] = new double[] { 5.7, 2.9, 4.2, 1.3, 0, 1, 0 };
            DataSet[97] = new double[] { 6.2, 2.9, 4.3, 1.3, 0, 1, 0 };
            DataSet[98] = new double[] { 5.1, 2.5, 3.0, 1.1, 0, 1, 0 };
            DataSet[99] = new double[] { 5.7, 2.8, 4.1, 1.3, 0, 1, 0 };
            //İRİS-VİRGİNİCA
            DataSet[100] = new double[] { 6.3, 3.3, 6.0, 2.5, 1, 0, 0 };
            DataSet[101] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
            DataSet[102] = new double[] { 7.1, 3.0, 5.9, 2.1, 1, 0, 0 };
            DataSet[103] = new double[] { 6.3, 2.9, 5.6, 1.8, 1, 0, 0 };
            DataSet[104] = new double[] { 6.5, 3.0, 5.8, 2.2, 1, 0, 0 };
            DataSet[105] = new double[] { 7.6, 3.0, 6.6, 2.1, 1, 0, 0 };
            DataSet[106] = new double[] { 4.9, 2.5, 4.5, 1.7, 1, 0, 0 };
            DataSet[107] = new double[] { 7.3, 2.9, 6.3, 1.8, 1, 0, 0 };
            DataSet[108] = new double[] { 6.7, 2.5, 5.8, 1.8, 1, 0, 0 };
            DataSet[109] = new double[] { 7.2, 3.6, 6.1, 2.5, 1, 0, 0 };
            DataSet[110] = new double[] { 6.5, 3.2, 5.1, 2.0, 1, 0, 0 };
            DataSet[111] = new double[] { 6.4, 2.7, 5.3, 1.9, 1, 0, 0 };
            DataSet[112] = new double[] { 6.8, 3.0, 5.5, 2.1, 1, 0, 0 };
            DataSet[113] = new double[] { 5.7, 2.5, 5.0, 2.0, 1, 0, 0 };
            DataSet[114] = new double[] { 5.8, 2.8, 5.1, 2.4, 1, 0, 0 };
            DataSet[115] = new double[] { 6.4, 3.2, 5.3, 2.3, 1, 0, 0 };
            DataSet[116] = new double[] { 6.5, 3.0, 5.5, 1.8, 1, 0, 0 };
            DataSet[117] = new double[] { 7.7, 3.8, 6.7, 2.2, 1, 0, 0 };
            DataSet[118] = new double[] { 7.7, 2.6, 6.9, 2.3, 1, 0, 0 };
            DataSet[119] = new double[] { 6.0, 2.2, 5.0, 1.5, 1, 0, 0 };
            DataSet[120] = new double[] { 6.9, 3.2, 5.7, 2.3, 1, 0, 0 };
            DataSet[121] = new double[] { 5.6, 2.8, 4.9, 2.0, 1, 0, 0 };
            DataSet[122] = new double[] { 7.7, 2.8, 6.7, 2.0, 1, 0, 0 };
            DataSet[123] = new double[] { 6.3, 2.7, 4.9, 1.8, 1, 0, 0 };
            DataSet[124] = new double[] { 6.7, 3.3, 5.7, 2.1, 1, 0, 0 };
            DataSet[125] = new double[] { 7.2, 3.2, 6.0, 1.8, 1, 0, 0 };
            DataSet[126] = new double[] { 6.2, 2.8, 4.8, 1.8, 1, 0, 0 };
            DataSet[127] = new double[] { 6.1, 3.0, 4.9, 1.8, 1, 0, 0 };
            DataSet[128] = new double[] { 6.4, 2.8, 5.6, 2.1, 1, 0, 0 };
            DataSet[129] = new double[] { 7.2, 3.0, 5.8, 1.6, 1, 0, 0 };
            DataSet[130] = new double[] { 7.4, 2.8, 6.1, 1.9, 1, 0, 0 };
            DataSet[131] = new double[] { 7.9, 3.8, 6.4, 2.0, 1, 0, 0 };
            DataSet[132] = new double[] { 6.4, 2.8, 5.6, 2.2, 1, 0, 0 };
            DataSet[133] = new double[] { 6.3, 2.8, 5.1, 1.5, 1, 0, 0 };
            DataSet[134] = new double[] { 6.1, 2.6, 5.6, 1.4, 1, 0, 0 };
            DataSet[135] = new double[] { 7.7, 3.0, 6.1, 2.3, 1, 0, 0 };
            DataSet[136] = new double[] { 6.3, 3.4, 5.6, 2.4, 1, 0, 0 };
            DataSet[137] = new double[] { 6.4, 3.1, 5.5, 1.8, 1, 0, 0 };
            DataSet[138] = new double[] { 6.0, 3.0, 4.8, 1.8, 1, 0, 0 };
            DataSet[139] = new double[] { 6.9, 3.1, 5.4, 2.1, 1, 0, 0 };
            DataSet[140] = new double[] { 6.7, 3.1, 5.6, 2.4, 1, 0, 0 };
            DataSet[141] = new double[] { 6.9, 3.1, 5.1, 2.3, 1, 0, 0 };
            DataSet[142] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
            DataSet[143] = new double[] { 6.8, 3.2, 5.9, 2.3, 1, 0, 0 };
            DataSet[144] = new double[] { 6.7, 3.3, 5.7, 2.5, 1, 0, 0 };
            DataSet[145] = new double[] { 6.7, 3.0, 5.2, 2.3, 1, 0, 0 };
            DataSet[146] = new double[] { 6.3, 2.5, 5.0, 1.9, 1, 0, 0 };
            DataSet[147] = new double[] { 6.5, 3.0, 5.2, 2.0, 1, 0, 0 };
            DataSet[148] = new double[] { 6.2, 3.4, 5.4, 2.3, 1, 0, 0 };
            DataSet[149] = new double[] { 5.9, 3.0, 5.1, 1.8, 1, 0, 0 };

            int trainDataCount = (DataSet.Length * 70) / 100;
            int testDataCount = (DataSet.Length * 30) / 100;

            Random r = new Random();

            int[] trainDataNumbers = new int[trainDataCount];
            for (int i = 0; i < trainDataNumbers.Length; i++)
            {
                trainDataNumbers[i] = r.Next(0, trainDataCount);
            }

            int[] testDataNumbers = new int[testDataCount];
            for (int i = 0; i < testDataNumbers.Length; i++)
            {
                testDataNumbers[i] = r.Next(0, testDataCount);
            }
            /*
            for (int i = 0; i < trainDataNumbers.Length; i++)
            {
                Console.WriteLine(i+"="+trainDataNumbers[i]);
            }

            for (int i = 0; i < testDataNumbers.Length; i++)
            {
                Console.WriteLine(i + "=" + testDataNumbers[i]); 
            }
            */
            double[][] trainData = new double[trainDataCount][];
            for (int j = 0; j < trainData.Length; j++)
            {
                trainData[j] = DataSet[trainDataNumbers[j]];
            }
            /*
            for(int a=0;a<trainData.Length;a++)
            {
                for(int b=0;b<DataSet[0].Length;b++)
                {
                    Console.Write("[]" + a + "[]" + b + "=" + trainData[a][b]+"\n");
                }
                Console.WriteLine();
            }*/

            double[][] testData = new double[testDataCount][];
            for (int j = 0; j < testData.Length; j++)
            {
                testData[j] = DataSet[testDataNumbers[j]];
            }
            /*
            for (int a = 0; a < testData.Length; a++)
            {
                for (int b = 0; b < DataSet[0].Length; b++)
                {
                    Console.Write("[]" + a + "[]" + b + "=" + testData[a][b] + "\n");
                }
                Console.WriteLine();
            }*/

            int iNeuronCount = 4;
            int hNeuronCount = 4;
            int oNeuronCount = 3;

            double[] input = new double[4];
            double[] output = new double[3];
            double[] target = new double[3];
            double[] hidden = new double[4];

            double[][] ihWeihgts = new double[4][];
            double[] hBiases = new double[4];
            double[][] hoWeihgts = new double[4][];
            double[] oBiases = new double[3];

            int tmp = 0;
            for (int i = 0; i < iNeuronCount; i++)
            {
                ihWeihgts[i] = new double[4];
                for (int j = 0; j < hNeuronCount; j++)
                {
                    ihWeihgts[i][j] = BestAnimalValues[tmp];
                    tmp++;
                }
            }//ih agırlıklar
            for (int i = 0; i < hNeuronCount; i++)
            {
                hBiases[i] = BestAnimalValues[tmp];
                tmp++;
            }//hidden biası

            for (int i = 0; i < hNeuronCount; i++)
            {
                hoWeihgts[i] = new double[4];
                for (int j = 0; j < oNeuronCount; j++)
                {
                    hoWeihgts[i][j] = BestAnimalValues[tmp];
                    tmp++;
                }
            }

            for (int i = 0; i < oNeuronCount; i++)
            {
                oBiases[i] = BestAnimalValues[tmp];
                tmp++;
            }

            double[][] Ttargets = new double[testData.Length][];
            double[][] Toutputs = new double[testData.Length][];

            //MSE Hesaplanarak Fit Değerleri bulunacak
            for (int i = 0; i < testData.Length; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    input[j] = testData[i][j];
                }
                int sayac = 0;
                for (int m = 4; m < testData[i].Length; m++)
                {
                    target[sayac++] = testData[i][m];
                }

                double[] hSum = new double[hNeuronCount];

                for (int j = 0; j < hNeuronCount; j++)
                {
                    for (int k = 0; k < iNeuronCount; k++)
                    {
                        hSum[j] += input[k] * ihWeihgts[k][j]; //ara katmandaki nöronların değerlerini hesapladık.
                    }
                    hSum[j] += hBiases[j];
                    hSum[j] = Sigmoid(hSum[j]);
                }

                double[] oSum = new double[oNeuronCount];

                for (int j = 0; j < oNeuronCount; j++)
                {
                    for (int k = 0; k < hNeuronCount; k++)
                    {
                        oSum[j] += hSum[k] * hoWeihgts[k][j]; //çıkış katmanındaki nöronlarımızın değerlerini hesapladık.
                    }
                    oSum[j] += oBiases[j];
                    output[j] = Sigmoid(oSum[j]);
                }


                Ttargets[i] = new double[testData.Length];
                for (int j = 0; j < target.Length; j++)
                {
                    Ttargets[i][j] = target[j];
                }


                Toutputs[i] = new double[testData.Length];
                for (int j = 0; j < output.Length; j++)
                {
                    Toutputs[i][j] = output[j];
                }
            }

            double MSE = 0;
            double M = 0;
            for (int u = 0; u < Ttargets.Length; u++)
            {
                for (int w = 0; w < Ttargets[0].Length; w++)
                {
                    MSE += Math.Pow((Ttargets[u][w] - Toutputs[u][w]), 2);
                    M += Ttargets[u][w];
                }
            }
            double pay = MSE;
            MSE /= Ttargets.Length;
            M /= Ttargets.Length;
            double payda = 0;
            for (int u = 0; u < Ttargets.Length; u++)
            {
                for (int w = 0; w < Ttargets[0].Length; w++)
                {
                    payda += Math.Pow((Ttargets[u][w] - M), 2);
                }
            }
            double R2 = 1 - (pay / payda);

            Console.WriteLine("Test Datalarının Doğruluğu = " + R2);
        }

        private static double Sigmoid(double v)
        {
            return 1 / (1 + (Math.Pow(Math.E, v * -1)));
        }
    }
}
