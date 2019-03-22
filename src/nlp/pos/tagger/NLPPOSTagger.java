/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nlp.pos.tagger;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.io.*;
import java.util.*;

import edu.stanford.nlp.io.*;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.*;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

/**
 *
 * @author kksaikrishna
 */
public class NLPPOSTagger {

    /**
     * @param args the command line arguments
     */
    
    HashMap<String,HashMap<String,Integer>> countTree = new HashMap<String,HashMap<String,Integer>>();
    HashMap<String,Integer> trainedTagCounts = new HashMap<String,Integer>();
    ArrayList<String> tagSequence = new ArrayList<String>();
    String LikelihoodProbabilities = new String();
    String TagBigramProbabilities = new String();
    
    void train() throws FileNotFoundException
    {
        String entireFileText = new Scanner(new File("training.txt")).useDelimiter("\\A").next();
        entireFileText = entireFileText.toLowerCase();
       // System.out.println(entireFileText);

        Properties props = new Properties();
        props.setProperty("annotators","tokenize, ssplit, pos");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        Annotation annotation = new Annotation(entireFileText);
        pipeline.annotate(annotation);
        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        for (CoreMap sentence : sentences) 
        {
            for (CoreLabel token: sentence.get(CoreAnnotations.TokensAnnotation.class))
            {
                String word = token.get(CoreAnnotations.TextAnnotation.class);
                // this is the POS tag of the token
                String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                //System.out.println(word + "/" + pos);
                addWordTagCount(word,pos);
                addTagCount(pos);
                tagSequence.add(pos);
            }
        }
    }
    
    void addWordTagCount(String word, String tag)
    {
        if(!countTree.containsKey(word))
        {
            HashMap<String,Integer> tagCount = new HashMap<String,Integer>();
            tagCount.put(tag, 1);
            countTree.put(word, tagCount);
        }
        else
        {
            HashMap<String,Integer> tagCount = countTree.get(word);
            if(!tagCount.containsKey(tag))
            {
                tagCount.put(tag, 1);
            }
            else
            {
                tagCount.put(tag, tagCount.get(tag)+1);
            }
            countTree.put(word, tagCount);
        }
    }
    
    void addTagCount(String tag)
    {
        if(!trainedTagCounts.containsKey(tag))
        {
            trainedTagCounts.put(tag, 1);
        }
        else
        {
            trainedTagCounts.put(tag, trainedTagCounts.get(tag)+1);
        }
    }
    
    void getInput() throws FileNotFoundException
    {
        String entireFileText = new Scanner(new File("input.txt")).useDelimiter("\\A").next();
        System.out.println("\nTest Sentence: "+ entireFileText + "\n\n");
        entireFileText = entireFileText.toLowerCase();
        
        ArrayList<String> prTags = new ArrayList<String>();
        
        Reader r = new StringReader(entireFileText);
        PTBTokenizer<Word> tokenizer = PTBTokenizer.newPTBTokenizer(r);
        
        while (tokenizer.hasNext()) 
        {
            Word w = tokenizer.next();
            ArrayList<String> tags = getTags(w.toString());
            for(int i = 0; i < tags.size(); i++)
            {
                String tag = tags.get(i);
                calculateLikelihoodProbabilities(w.toString(),tag);
                
                if(!prTags.isEmpty())
                {
                    for(int j = 0; j<prTags.size(); j++)
                    {
                        calculateTagBigramProbabilities(tag, prTags.get(j));
                    }
                }
            }
            prTags = tags;
        }
        
       
    }
    
    ArrayList<String> getTags(String word)
    {
        ArrayList<String> tags = new ArrayList<String>();
        
        if(countTree.containsKey(word))
        {
            for(String tag : countTree.get(word).keySet())
            {
                tags.add(tag);
            }
           
            return tags;
        }
        else
        {
            return null;
        }
    }
    
    int getCount(String word, String tag)
    {
        if(countTree.containsKey(word))
        {
            if(countTree.get(word).containsKey(tag))
            {
                return countTree.get(word).get(tag);
            }
            else
            {
                return 0;
            }
        }
        else
        {
            return 0;
        }
        
    }
    
    int getTagCount(String tag)
    {
        if(trainedTagCounts.containsKey(tag))
            return trainedTagCounts.get(tag);
        else
            return 0;
    }
    
    int getTagSequenceCount(String curTag, String prTag)
    {
        int count = 0;
        
        for(int i = 0; i<tagSequence.size();i++)
        {
            if(tagSequence.get(i).equalsIgnoreCase(prTag))
            {
                if(tagSequence.get(++i).equalsIgnoreCase(curTag))
                {
                    count++;
                }
            }
        }
        
        return count;
    }
    
    void calculateLikelihoodProbabilities(String word, String tag)
    {
        long count = getCount(word, tag);
        long tagCount = getTagCount(tag);
        
        StringBuilder sbuf = new StringBuilder();
        String temp = new String();

        Formatter fmt = new Formatter(sbuf);
        fmt.format("%.5f\n", (double)count/(double)tagCount);
        temp = fmt.toString();
        
        LikelihoodProbabilities += "P(" + word + "|" + tag + ") = C(" 
                + word + "," + tag + ")/C(" + tag + ") = " 
                + count + "/" + tagCount + " = " + temp;
        
    }
    
    void calculateTagBigramProbabilities(String curTag, String prTag)
    {
        long count = getTagSequenceCount(curTag,prTag);
        long tagCount = getTagCount(prTag);
        
        
        StringBuilder sbuf = new StringBuilder();
        String temp = new String();

        Formatter fmt = new Formatter(sbuf);
        fmt.format("%.5f\n", (double)count/(double)tagCount);
        temp = fmt.toString();
        
        TagBigramProbabilities += "P(" + curTag + "|" + prTag + ") = C(" 
                + prTag + "," + curTag + ")/C(" + prTag + ") = " 
                + count + "/" + tagCount + " = " + temp;
    }
    
    void printProbabilities()
    {
        System.out.print("********Likelihood Probabilities********\n"+LikelihoodProbabilities
                + "\n******** Tag Bigram Probabilities ********\n" + TagBigramProbabilities);
        
    }
    
    public static void main(String[] args) throws FileNotFoundException {
        
       NLPPOSTagger objNLPPOSTagger = new NLPPOSTagger();
       objNLPPOSTagger.train();
       objNLPPOSTagger.getInput();
       objNLPPOSTagger.printProbabilities();
    }
    
}


