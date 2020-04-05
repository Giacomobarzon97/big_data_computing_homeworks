import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

public class G48HW1 {
    public static void main(String[] args) throws IOException {
        //SPARK CONFIGURATION
        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }
        SparkConf conf = new SparkConf(true).setAppName("Homework1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");
        // INPUT READING
        int K = Integer.parseInt(args[0]);
        JavaRDD<String> pairString = sc.textFile(args[1]).repartition(K);
        // SETTING GLOBAL VARIABLES
        long numpairs,numclasses;
        numpairs = pairString.count();
        JavaPairRDD<String, Long> count;


        //FIRST EXERCISE
        System.out.println("VERSION WITH DETERMINISTIC PARTITION");
        Random randomGenerator = new Random();
        count = pairString
            .flatMapToPair((pair) -> {    // <-- MAP PHASE (R1)
                //INPUT:Pair(i, yi) where is an incremental index starting from 0
                //And yi is a category(String) Associated with that index

                int index=Integer.parseInt(pair.split(" ")[0]);
                String category=pair.split(" ")[1];

                ArrayList<Tuple2<Integer, String>> pairs = new ArrayList<>();
                pairs.add(new Tuple2<>(index%K,category));
                return pairs.iterator();

                //OUTPUT:Single pair (j,yi) where j is an index bounded between
                //0 and K
            })
            .groupByKey()    // <-- REDUCE PHASE (R1)
            .flatMapToPair((pair) -> {
                //INPUT: For each key j between 0 and K gather the set Sj
                //all intermediate pairs with key j

                HashMap<String, Long> counts = new HashMap<>();
                for (String  s: pair._2()) {
                    counts.put(s,counts.getOrDefault(s,0L)+1);
                }
                ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                for (Map.Entry<String, Long> e : counts.entrySet()) {
                    pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                }
                return pairs.iterator();

                //OUTPUT: for each class y labeling some object in Sj
                //produce the pair (y, cj(y)), where cj(y) is the
                //number of objects of Sj labeled with y.

            }).groupByKey()// <-- REDUCE PHASE (R2)
            .flatMapToPair((pair) -> {
                //INPUT:: for each class y, gather the at most K pairs (y, cj(y))
                // resulting at the end of the previous round

                Long sum=0L;
                for (Long  i: pair._2()) {
                    sum=sum+i;
                }
                ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                pairs.add(new Tuple2<String,Long>(pair._1(),sum));
                return pairs.iterator();

                //OUTPUT:return the output pair (γ,sum(cj(y)for each j)

            });
        String res="Output pairs = ";
        for(Tuple2<String,Long> tuple:count.sortByKey().collect()) {
            res=res+"("+tuple._1()+","+tuple._2()+"),";
        }
        System.out.println(res.substring(0,res.length()-1)+";");

        //Second Exercise
        System.out.println("VERSION WITH SPARK PARTITIONS");
        count = pairString
            .flatMapToPair((pair) -> {    // <-- MAP PHASE (R1)
                //INPUT:Pair(i, yi) where is an incremental index starting from 0
                //And yi is a category(String) Associated with that index

                int index=Integer.parseInt(pair.split(" ")[0]);
                String category=pair.split(" ")[1];
                ArrayList<Tuple2<Integer, String>> pairs = new ArrayList<>();
                pairs.add(new Tuple2<>(index%K,category));
                return pairs.iterator();

                //OUTPUT:Single pair (j,yi) where j is an index bounded between
                //0 and K.
            })
            .mapPartitionsToPair((wc) -> {    // <-- REDUCE PHASE (R1)
                //INPUT:: for each class y, gather the at most K pairs (y, cj(y))
                // resulting at the end of the previous round

                HashMap<String, Long> counts = new HashMap<>();
                Long num_pairs=0L;
                while (wc.hasNext()){
                    num_pairs++;
                    Tuple2<Integer, String> tuple = wc.next();
                    counts.put(tuple._2(),counts.getOrDefault(tuple._2(),0L)+1);
                }
                 ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                for (Map.Entry<String, Long> e : counts.entrySet()) {
                    pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                }
                pairs.add(new Tuple2<>("maxPartitionSize", num_pairs));
                return pairs.iterator();

                //OUTPUT: for each class y labeling some object in Sj
                //produce the pair (y, cj(y)), where cj(y) is the
                //number of objects of Sj labeled with y.
                //Moreover produce the pair ("maxPartitionSize", partition_dimension) for each partition j where
                // partition_dimension is the number of pairs associated with that partition
            })
            .groupByKey()     // <-- REDUCE PHASE (R2)
            .flatMapToPair((pair) -> {
                //INPUT:: for each class y, gather the at most K pairs (y, cj(y))
                // resulting at the end of the previous round
                //For the special case where y="maxPartitionSize" the program gathers as well
                //at most K pairs

                //Checking the type of the pair
                if (!pair._1().equals("maxPartitionSize")) {
                    //For each class y producing a single
                    // pair witch contains the sum of all intermediate pairs with class yj
                    Long sum = 0L;
                    for (Long i : pair._2()) {
                        sum = sum + i;
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    pairs.add(new Tuple2<String, Long>(pair._1(), sum));
                    return pairs.iterator();
                }else{
                    //Calculating max betweeen all partition_sizes
                    Long max=0L;
                    for (Long i : pair._2()) {
                        if (i>max){
                            max=i;
                        }
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    pairs.add(new Tuple2<String, Long>("maxPartitionSize", max));
                    return pairs.iterator();
                }
                //OUTPUT: return the output pair (γ,sum(cj(y)for each j) plus
                //one pair of the type ("maxPartitionSize",maximum_partition_dimension) where
                //maximum_partition_dimension is the maximum number of pairs assigned to a partition
            });
        List<Tuple2<String,Long>> list=count.filter((pair)->{
            return !pair._1().equals("maxPartitionSize");
        }).sortByKey().collect();
        Tuple2<String,Long> most_frequent=new Tuple2<String,Long>("",-1L);
        for(Tuple2<String,Long> i: list){
            if(i._2()>most_frequent._2()){
                most_frequent=i;
            }
        }
        System.out.println("Most frequent class = "+"("+most_frequent._1()+","+most_frequent._2()+")");

        Long max_partion_size=count.filter((pair)->{
            return pair._1().equals("maxPartitionSize");
        }).sortByKey().collect().get(0)._2();
        System.out.println("Max partition size = "+""+ max_partion_size);
    }
}
