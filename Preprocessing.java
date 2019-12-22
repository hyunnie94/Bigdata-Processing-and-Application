package hadoop;

import java.io.IOException;
import java.util.Iterator;

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

public class Preprocessing {

    public static class Map extends MapReduceBase implements
            Mapper<LongWritable, Text, Text, Text> {
    	
    	private Text user = new Text();
    	
        @Override
        public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter)
                throws IOException {
        	
            String line = value.toString();
            String[] split_line= line.split(" ");
            
            user.set(split_line[0]);
            output.collect(user, new Text(split_line[split_line.length-1]));            

        }
    }

    public static class Reduce extends MapReduceBase implements
            Reducer<Text,Text,LongWritable,Text> {

        @Override
        public void reduce(Text key, Iterator<Text> values,
                OutputCollector<LongWritable,Text> output, Reporter reporter)
                throws IOException {
            String tmp = "";
            int count = 0;
            while (values.hasNext()) {
                tmp = tmp+ values.next().toString()+",";
                count++;
            }
            String temp = key.toString()+","+tmp.substring(0, tmp.length()-1);          
            output.collect(new LongWritable(count), new Text(temp));
        }
    }
 
   
    public static void main(String[] args) throws Exception {

        JobConf conf = new JobConf(Preprocessing.class);
        conf.setJobName("Preprocessing");

        conf.setMapOutputKeyClass(Text.class);
        conf.setMapOutputValueClass(Text.class);
        conf.setOutputKeyClass(LongWritable.class);
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

