package hadoop;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;

public class BusCount {

    public static class Map extends MapReduceBase implements
            Mapper<LongWritable, Text, Text, Text> {
    	
    	private Text user = new Text();
    	
        @Override
        public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter)
                throws IOException {

            String line = value.toString();
            String[] split_line= line.split("\t");
            String[] Info = split_line[1].split(",");
            
            user.set(Info[0]);
            
            for(int i=1; i<Info.length; i++) {
            	output.collect(new Text(Info[i]),user);
            }

        }
    }

    public static class Reduce extends MapReduceBase implements
            Reducer<Text,Text,LongWritable,Text> {
    	static long count =0;
    	static HashMap<Text,String> map1 = new HashMap<>();
    	
        @Override
        public void reduce(Text key, Iterator<Text> values,
                OutputCollector<LongWritable,Text> output, Reporter reporter)
                throws IOException {      
        	
        	map1.put(key, "(");
        	Set<Text> set = new HashSet<>();
           while (values.hasNext()) {
        	   Text temp = values.next();
        	   if(!set.contains(temp)) {
        		   set.add(temp);
        		   map1.put(key, map1.get(key)+temp.toString()+",");
        	   } 	
            }
           String line = map1.get(key);
           map1.put(key,line.substring(0, line.length()-1)+")");

        	 count++;
        	 long len = line.length();
        	 output.collect(new LongWritable(len), new Text(key.toString()+"  "+map1.get(key)));        	   
           
        }
    }

    public static void main(String[] args) throws Exception {

        JobConf conf = new JobConf(BusCount.class);
        conf.setJobName("BusCount");

        conf.setOutputKeyClass(Text.class);
        conf.setOutputValueClass(Text.class);

        conf.setMapperClass(Map.class);
        conf.setReducerClass(Reduce.class);

        conf.setInputFormat(TextInputFormat.class);
        conf.setOutputFormat(TextOutputFormat.class);

        FileInputFormat.setInputPaths(conf, new Path(args[0]));
        FileOutputFormat.setOutputPath(conf, new Path(args[1]));

        JobClient.runJob(conf);

    }
}

