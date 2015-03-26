using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace StockNet
{
    /* Stock predictor
     * File location: http://www.google.com/finance/getprices?i=[freq]&p=[days]d&f=d,o,h,l,c,v&df=cpct&q=[name]
     * Columns: DATE,CLOSE,HIGH,LOW,OPEN,VOLUME
     */

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        // Neural Network
        NeuralNetwork[] networks;
        double[] errors;
        int INPUTNUM;
        int HIDDENNUM;
        const int OUTPUTNUM = 1;

        // Genetic Algorithm
        const int POPULATION = 64;
        const int GEN = 1000;
        const int ELITE = 8;
        const double MUTATION = 0.008;

        // Data
        double[][] aapl_day, dow_day, snp_day, nasdaq_day;
        double[][] aapl, dow, snp, nasdaq;
        const int TICKS = 390;
        const int SAMPLE = 64;
        const int DATA_LEN = 5;
        const int WIN_DAY = 12;
        const int WIN_MIN = 12;

        const int NORMALIZE = 200;
        Random random = new Random();

        private void StartTrain(object sender, RoutedEventArgs e)
        {
            INPUTNUM =
                1 +                     // Day of the week
                1 +                     // Minute of the day
                (WIN_DAY * DATA_LEN +   // data for past WIN_DAY days
                WIN_MIN * DATA_LEN +    // data for past WIN_MIN minutes
                1) *                    // open for the day
                1;                      // DOW, S&P, NASDAQ
            HIDDENNUM = (int)(INPUTNUM * 2 / 3);

            // Initialize networks
            networks = new NeuralNetwork[POPULATION];
            for (int i = 0; i < POPULATION; i++)
            {
                networks[i] = new NeuralNetwork(INPUTNUM, HIDDENNUM, OUTPUTNUM, random);
            }
            errors = new double[POPULATION];

            // Begin training
            for (int gen = 0; gen < GEN; gen++)
            {
                // Pick a random day to sample
                string path = "../../Data/";
                string[] files = Directory.GetFiles(path + "AAPL/");
                int day = random.Next(0, files.Length - 1);

                // Pick a minute range to sample
                int start = random.Next(WIN_MIN, TICKS - SAMPLE);

                // Load data
                StreamReader sr = new StreamReader(files[day]);
                string text = sr.ReadToEnd();
                sr.Close();

                // Parse data
                aapl = new double[DATA_LEN][];
                for (int i = 0; i < DATA_LEN; i++) { aapl[i] = new double[TICKS]; }
                string[] lines = text.Split('\n');
                for (int i = 0; i < TICKS; i++)
                {
                    string[] line = lines[i].Split(',');
                    for (int j = 0; j < DATA_LEN; j++)
                    {
                        aapl[j][i] = Convert.ToDouble(line[j + 1]);
                    }
                }

                // Normalize data
                for (int i = 0; i < DATA_LEN; i++)
                {
                    double max = aapl[i].Max();
                    double min = aapl[i].Min();
                    double range = max - min;
                    aapl[i] = aapl[i].Select(n => (n - min) / range).ToArray<double>();
                }

                sr = new StreamReader(path + "AAPL/daily.txt");
                text = sr.ReadToEnd();
                sr.Close();

                // Parse day data
                aapl_day = new double[DATA_LEN][];
                for (int i = 0; i < DATA_LEN; i++) { aapl_day[i] = new double[WIN_DAY]; }
                lines = text.Split('\n');
                for (int i = 0; i < WIN_DAY; i++)
                {
                    string[] line = lines[i + files.Length - 1 - day].Split(',');
                    for (int j = 0; j < DATA_LEN; j++)
                    {
                        aapl_day[j][WIN_DAY - 1 - i] = Convert.ToDouble(line[j + 1]);
                    }
                }

                // Normalize day data
                for (int i = 0; i < DATA_LEN; i++)
                {
                    double max = aapl_day[i].Max();
                    double min = aapl_day[i].Min();
                    double range = max - min;
                    aapl_day[i] = aapl_day[i].Select(n => (n - min) / range).ToArray<double>();
                }

                // Test each network in the population
                for (int i = 0; i < POPULATION; i++)
                {
                    errors[i] = 0;
                    // Test each window of size WIN_MIN
                    for (int j = start; j < start + SAMPLE; j++) //TICKS
                    {
                        /* Fill in the input values */
                        int c = 0;

                        // Day and minute
                        networks[i].inputValues[c++] = (day % 5) / 5.0;
                        networks[i].inputValues[c++] = (double)j / TICKS;

                        // Data for past WIN_DAY days
                        for (int k = 0; k < WIN_DAY; k++)
                            for (int l = 0; l < DATA_LEN; l++)
                                networks[i].inputValues[c++] = aapl_day[l][k];

                        // Data for past WIN_MIN minutes
                        for (int k = j - WIN_MIN; k < j; k++)
                            for (int l = 0; l < DATA_LEN; l++)
                                networks[i].inputValues[c++] = aapl[l][k];

                        // Forward propagate
                        networks[i].FeedForward();

                        // Get error
                        double a = networks[i].outputValues[0];
                        double b = aapl[0][j];
                        errors[i] += Math.Abs(networks[i].outputValues[0] - aapl[0][j]) / networks[i].outputValues[0];
                    }
                    errors[i] = errors[i] * 100 / SAMPLE;
                    networks[i].fitness = errors[i];
                }

                // Genetic Algorithm
                GeneticAlgorithm();

                string write = errors.Min().ToString();
                Console.WriteLine(write);
            }
        }

        public void GeneticAlgorithm()
        {
            Array.Sort(errors, networks);
            //Array.Reverse(networks);

            for (int i = ELITE; i < POPULATION; i++)
            {
                // Pick two parents
                int parent1 = random.Next(0, ELITE);
                int parent2 = random.Next(0, ELITE);

                // Pick two splice points
                int geneNum = HIDDENNUM * (INPUTNUM + 1) + OUTPUTNUM * (HIDDENNUM + 1);
                int splice1 = random.Next(0, (int)(geneNum / 2));
                int splice2 = splice1 + (int)(geneNum / 2);

                // Get parent genes
                double[] genes1 = networks[parent1].FlattenWeights();
                double[] genes2 = networks[parent2].FlattenWeights();

                // Splice genes
                double[] cgenes = new double[geneNum];
                double gene;
                for (int j = 0; j < geneNum; j++)
                {
                    gene = random.NextDouble();
                    if (gene < MUTATION)
                        cgenes[j] = gene;
                    else if (j < splice1 || j >= splice2)
                        cgenes[j] = genes1[j];
                    else
                        cgenes[j] = genes2[j];
                }

                // Inflate genes
                networks[i].InflateGenes(cgenes);
            }
        }
    }
}
