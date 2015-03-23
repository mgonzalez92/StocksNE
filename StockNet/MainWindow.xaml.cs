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
        const int INPUTNUM = 15;
        const int HIDDENNUM = 10;
        const int OUTPUTNUM = 1;

        // Genetic Algorithm
        const int POPULATION = 200;
        const int GEN = 1000;
        const int ELITE = 10;
        const double MUTATION = 0.01;

        // Data
        double[] data;
        const int TICKS = 391;
        const int DAYS = 10;

        const int NORMALIZE = 200;
        Random random = new Random();

        private void StartTrain(object sender, RoutedEventArgs e)
        {
            // Initialize networks
            networks = new NeuralNetwork[POPULATION];
            for (int i = 0; i < POPULATION; i++)
            {
                networks[i] = new NeuralNetwork(INPUTNUM, HIDDENNUM, OUTPUTNUM, random);
            }
            errors = new double[POPULATION];

            // Load data
            StreamReader sr = new StreamReader("C:/Users/costco0371/Documents/Visual Studio 2013/Projects/StockNet/StockNet/data.txt");
            string text = sr.ReadToEnd();
            sr.Close();

            // Parse data
            data = new double[DAYS * TICKS];
            string[] lines = text.Split('\n');
            for (int i = 0; i < lines.Length; i++)
            {
                string[] line = lines[i].Split(',');
                data[i] = Convert.ToDouble(line[1]);
            }

            // Begin training
            for (int gen = 0; gen < GEN; gen++)
            {
                // Pick a random day to sample
                int day = random.Next(0, DAYS);

                // Test each network in the population
                for (int i = 0; i < POPULATION; i++)
                {
                    errors[i] = 0;
                    // Test each window of size INPUTNUM
                    for (int j = INPUTNUM; j < TICKS; j++)
                    {
                        // Fill the input values
                        for (int k = j - INPUTNUM; k < j; k++)
                            networks[i].inputValues[k - j + INPUTNUM] = data[day * TICKS + k] / NORMALIZE;

                        // Forward propagate
                        networks[i].FeedForward();

                        // Get error
                        double a = networks[i].outputValues[0] * NORMALIZE;
                        double b = data[day + j];
                        errors[i] += Math.Abs(networks[i].outputValues[0] * NORMALIZE - data[day * TICKS + j]) / data[day * TICKS + j];
                    }
                    networks[i].fitness = errors[i];
                }

                // Genetic Algorithm
                GeneticAlgorithm();

                string write = "gen: " + gen + ", error: " + errors.Min();
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
