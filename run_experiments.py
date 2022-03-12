# coding: utf-8
#
# Filename: run_experiments.py
# Author: Reto Gubelmann
# Date modified: 2022/03/12
# Input files needed: Yes
#
# (c) 2022 Reto Gubelmann
#
#************************************************************
# Functionality
#
# The Script takes the following arguments as input:
# - A taskfile.csv with tasks in a certain form
# - A femalename.txt with female first names
# - A malename.txt with male first names
# - A professions.txt with a list of professions with determiners
# - A masteroutput.csv to write output to
# - A modelmatters.csv listing Specifications of modeltypes
# - Specification of cuda nr.
# - Specification of Beginning of taskline (real line +1)
# - Specification of End of taskline (real line +1)   
#
# The script then does the following:
# 1. It opens input- and masteroutputfile 
#    names, professions, modelmatters, task-details into 2d-arrays each (1 per taskdetail)
# 2. Iterates over numerous loops:
#     for model in modelarray (from modelmatters)
#          for schema in schema-array (from taskfile)
#                open fulloutputfile for writing
#                start counters measuring exactly wrong
#                for name in name-array (two loops, one for female and one for male, if needed)
#                     for profession in profession-array (from professions)
#                            do predict mask in:
#                            FNAME is PROF and she likes to [MASK].
#                            obtain tokens of respective rank from [MASK]
#                            build sentence according to schema, name, prof, and tokens
#                            let the model predict
#                            update, etc.
##
# IMPORTANT: Two Things have to be set by specifying variables in the script, just search for "IMPORTANT":
# - Probrank, specified by toporbottom
# - Outputfiledirektory
#
#*************************************************************
# Sample call
#
# (remember to specify output-directory. It is now TESTING)
#
# # Date modified: 2022/11/26
#python3 run_experiments.py exp3_kleopatra_ddolby.csv Input/top100fnames.txt Input/top100mnames.txt Input/professions.txt \
    # master_dolbyruns_prob0_exp3_task1bis3_all.csv Input/modelmatters-complete.csv 1 1 3 &

#
#*************************************************************
# 
# 
#Begin of Program

# Import Modules, etc.

import re
import sys
import torch as torch
from transformers import (AutoTokenizer,  AutoModelForCausalLM, AutoModelForMaskedLM)
from timeit import default_timer as timer
import argparse

# Parse arguments

parser = argparse.ArgumentParser()

parser.add_argument("taskfile", type=str,
                    help="specify the file to extract the tasks from; see folder Input for sample")

parser.add_argument("fnamefile", type=str,
                    help="specify a filename with female first names; see folder Input for sample")
parser.add_argument("mnamefile", type=str,
                    help="specify a filename with male first names")


parser.add_argument("professionsfile", type=str,
                    help="specify a filename containing a list of professions with determines")

parser.add_argument("masteroutputfile", type=str,
                    help="specify a filename to write the output to")


parser.add_argument("modelfile", type=str,
                    help="specify the file to extract model information from")

parser.add_argument("cudanumber", type=int,
                    help="specify the cuda number to use")

parser.add_argument("begintaskline", type=int,
                    help="specify the task line from which to start this instance (number 0-> Line 1 (ok, da first line header))")

parser.add_argument("endtaskline", type=int,
                    help="specify the task line at which to end this instance (number 0-> Line 1 (ok, da first line header))")



args = parser.parse_args()

filename_fnamefile=args.fnamefile
filename_mnamefile=args.mnamefile
filename_professionsfile=args.professionsfile
filename_masteroutputfile=args.masteroutputfile
filename_modelfile=args.modelfile
filename_taskfile=args.taskfile


cudanumber=args.cudanumber
cudastring="cuda:"+str(cudanumber)
begintaskline=args.begintaskline
endtaskline=args.endtaskline

#print("Using cudanumberstring:",cudastring)


thefnamefile=open(filename_fnamefile,"r")
themnamefile=open(filename_mnamefile,"r")
professionsfile=open(filename_professionsfile, "r")

themasteroutputfile=open(filename_masteroutputfile,"w",buffering=1)
print("#ofACT;Negtypes;Conjunct;AddEls;MaskorFem;Probrank;Modelnames;EW-PM-Abs",end =" ; ",  file=themasteroutputfile)
print("EW-PM-Rel;EW-Tok-Abs;EW-Tok-Rel;Lines-Proced;CompTime;Schema",  file=themasteroutputfile)


themodelfile=open(filename_modelfile,"r")
next(themodelfile)
thetaskfile=open(filename_taskfile,"r")
#next(thetaskfile)

all_tasklines = thetaskfile.readlines()
#print(all_tasklines[4])

currtaskarray=[]

for i in range(begintaskline,endtaskline+1):
    currtaskarray.append(all_tasklines[i])
    
print("Currtaskarray is now:",currtaskarray)
    
# Read model specifica from file

modeltypes=[]
modelnames=[]
tokenizers=[]
printmodelnames=[]
modelpattern=re.compile('([^,]+),([^,]+),([^,]+),([^,]+)')
for line in themodelfile:
    matched=modelpattern.match(line)
    modeltypes.append(matched.group(1))
    modelnames.append(matched.group(2))
    tokenizers.append(matched.group(3))
    printmodelnames.append(matched.group(4))
    
# print("Modelmatters:",modeltypes,modelnames,tokenizers,printmodelnames)
    


# Read task specifics into arrays from file

numbofnegs=[]
negtypes=[]
conjunctions=[]
additionalels=[]
morf=[]
schemata=[]

taskpattern=re.compile('([^;]+);([^;]+);([^;]+);([^;]+);([^;]+);([^;]+)')
for line in currtaskarray:
    matched=taskpattern.match(line)
    numbofnegs.append(matched.group(1))
    negtypes.append(matched.group(2))
    conjunctions.append(matched.group(3))
    additionalels.append(matched.group(4))
    morf.append(matched.group(5))
    schemata.append(matched.group(6))
    
# print("Taskmatters:",numbofnegs,negtypes,conjunctions,additionalels,morf,schemata)    


# Initialize toporbottom-array

toporbottoms=[0] # IMPORTANT: specify probability rank of ACT-Tokens to be used here!




#############################################################################################################

## DEFINITION OF FUNCTIONS

# Initialize Arrays with predefined tokens to mask and exwrong, for de and en

def build_array(sourcefile):
    """"
    The function builds an array out of an inputfile,
    one array entry per line
    """

    array_in_construction=[]
    for line in sourcefile:
        cleanline=line.strip('\n ')
#        print(cleanline,file=sys.stderr)
        array_in_construction.append([cleanline])    
    return array_in_construction

## Function to extract top 3 predicted token in a given sentence with a masked token

def extract_top3token(rawline,ext_linecounter,ext_tokenizer,ext_model):
    """ This function takes in a sentence with a mask-token
    (as well as a number of other things), lets the model passed
    predict the token and returns an array with the top 3 predicted token.
    """
    # Tokenize input & Print
    int_tokenized_text = ext_tokenizer.tokenize(rawline)
    
#    print("\n\nLine N. _",ext_linecounter,"_ :",rawline,file=sys.stderr) STDERRCOMM
    print("\nLine N. _",ext_linecounter,"_ :",rawline,file=theoutputfile)
    
## Retrieve masked index & save top 3 token, probabilities & top 1 token

    if (cudaflag):
        input = ext_tokenizer.encode(rawline, return_tensors="pt").to(cudastring)
    else:
        input = ext_tokenizer.encode(rawline, return_tensors="pt")
    # Does what a tokenizer should be doing: tokenizes the input, returns
    # "numbers" of each token in pytorch tensor format for further processing.
#    print("Input sequence:",input,file=sys.stderr)
#    print("Mask token index:",ext_tokenizer.mask_token_id,file=sys.stderr)
    mask_token_index = torch.where(input == ext_tokenizer.mask_token_id)[1]
    # This line locates the mask_token_id (usually it seems to be 103) in the
    # encoded sequence, so an output of (tensor([0]), tensor([11]))
    # basically means that it is in the first (and only) row of encoded sequence
    # at position 11 (counting 0 as well); that's why we use [1]
    # we want to have the column entry, not the row entr.
    # Used to find the mask_token_index after having encoded the input as pt tensor.
    # "mask_token_id" is a system-variable of the tokenizer class.
#    torchwhere = torch.where(input == ext_tokenizer.mask_token_id)
#    print("Shape of torch.where output:",torchwhere,file=sys.stderr)
#    print("\tFirst dim of torch.where:",torchwhere[0],file=sys.stderr)
#    print("\tSecond dim of torch.where:",torchwhere[1],file=sys.stderr)
#    print("Shape of output tuple:",ext_model(input),file=sys.stderr)
    token_logits = ext_model(input)[0]
    # ext_model(input) returns a tuple of torch.FloatTensor depending on BertConfig
    # my config returns a logits tensor that has shape (batch size, sequence length, config.vocab_size).
    # see https://huggingface.co/transformers/model_doc/bert.html for details.
    # basic idea: for each sentence (dim=batch_size)
    # the model outputs the prediction scores for each token of the sentence
    # (dim=sequence_length) of each vocab entry (dim=vocab_size)
    # If specified, the model can also output hidden states or attentions.
    # This makes specification of the entry in the TUPLE necessary.
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Fetches the logits of the batch #0, at sequence position
    # mask_token_index, and the whole vocabulary there (:).

    top_3_token = torch.topk(mask_token_logits, 210, dim=1).indices[0].tolist()
    # mask_token_logits as input, 3 as number to be returned, dimension as dimension of
    # tensor along which to sort. As it returns another tensor, one has to then take the first row
    # and convert that row to a list.
    
    top_3_token_probs = torch.topk(torch.nn.functional.softmax(mask_token_logits, dim=1), 210, dim=1).values[0].tolist()
    # Same thing as above, but this time with the values being softmaxed, the dimension is again
    # the one along which you should gather values for softmaxing. 

    predicted_token=tokenizer.decode([top_3_token[0]])

    # Print out top 3 token and their probabilities,
    # store top 3 token in array.
    top3token=[]
    sum_probs=0
    int_diffflagsum=0
    int_diffflagind=0

    for toknumb in range(toporbottom,toporbottom+3):
        indexflag=0
        token=ext_tokenizer.decode([top_3_token[toknumb]])
        probability=top_3_token_probs[toknumb]
#        print(token,"\t",probability, file=sys.stderr) # STDERRCOMM
        print(token,"\t",probability, file=theoutputfile)
        top3token.append(token)
        sum_probs+=probability
#        if(probability> prob_threshold):
#            int_diffflagind=1
#    if(sum_probs> prob_threshold):
#        int_diffflagsum=1
#    else:
#        print("Top3token in sum too improbable, skipping, threshold:",prob_threshold, file=sys.stderr)
#        print("Top3token in sum too improbable, skipping, threshold:",prob_threshold, file=theoutputfile)
            
    return top3token, int_diffflagsum

## Function to predict a mask token in a given sentence and compare it with three exwrong token

def predict_check_excluded(ext_sequence, ext_tokenizer,ext_model,ext_currentmodel, ext_exwrong, numbacts):
    """
    This function first uses the model passed to predict a give nmasked token in a given sentence,
    then compares the top 10 predictions with the token passed in an array,
    prints out and updates counters accordingly. 
    """

    # Tokenize and print
    if (cudaflag):
        input = ext_tokenizer.encode(ext_sequence, return_tensors="pt").to(cudastring)
    else:
        input = ext_tokenizer.encode(ext_sequence, return_tensors="pt")
    
#    print("\nDIFFICULT INPUT:",ext_sequence,"\n",file=sys.stderr) # STDERRCOMM
    print("\nDIFFICULT INPUT (numbofacts:",numbacts,"):",ext_sequence,"\n",file=theoutputfile)
    mask_token_index = torch.where(input == ext_tokenizer.mask_token_id)[1]

    # Predict and store top 10 token with probabilities
    
    token_logits = ext_model(input)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_10_token = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist()
    top_10_token_probs = torch.topk(torch.nn.functional.softmax(mask_token_logits, dim=-1), 10, dim=1).values[0].tolist()
    predicted_token=tokenizer.decode([top_10_token[0]])

    # Compare predictions with passed token, print out
    # Update counters accordingly
    
    int_exactlywrongcounter=0
    int_exwrongweights=0
    for toknumb in range(0,10):
        indexflag=0
        token=ext_tokenizer.decode([top_10_token[toknumb]])
        probability=top_10_token_probs[toknumb]
#        try: # Changes made for enjoysrun: Matching of gerundiums.
#            ext_exwrong.index(token)
#        except ValueError:
#            indexflag=1
        exwrongtokencounter=0
        ewposition=0
        for exwrongtoken in ext_exwrong:
            exwrongtoken_cropped=exwrongtoken[:-1]
            if re.search(rf"^{exwrongtoken_cropped}[a-z]+", token, re.IGNORECASE):
                indexflag=1
                ewposition=exwrongtokencounter
                break
            exwrongtokencounter+=1
        
#        if(not indexflag):
        if(indexflag and ewposition < numbacts):
            int_exwrongweights+=probability
            if(toknumb==0):
                int_exactlywrongcounter+=1
                print(probability,"\t",token," ->FREW!",file=theoutputfile)
#                print(probability,"\t",token," ->FREW!",file=sys.stderr) #STDERRCOMM
            else:
                print(probability,"\t",token," ->EW!",file=theoutputfile)
#                print(probability,"\t",token," ->EW!",file=sys.stderr) # STDERRCOMM
        else:
            print(probability,"\t",token,file=theoutputfile)
#            print(probability,"\t",token, file=sys.stderr) # STDERRCOMM
        
            
            

    return int_exwrongweights, int_exactlywrongcounter

################################################################################

# Build arrays out of files

professionsarray=build_array(professionsfile)
fnamearray=build_array(thefnamefile)
mnamearray=build_array(themnamefile)


#print("Length of professionsarray is",len(professionsarray), file=sys.stderr)
#print("First ten items are:",professionsarray[:10], file=sys.stderr)    
    
#print("Length of fnamearray is",len(fnamearray), file=sys.stderr)
#print("First ten items are:",fnamearray[:10], file=sys.stderr)    

  
# Initializing Model loop
for modelline in range(len(modeltypes)):
    
    # Load tokenizer, set Variables, load model according to input
    tokenizer = AutoTokenizer.from_pretrained(tokenizers[modelline])
    


    if (modeltypes[modelline]=="bert"):
        model = AutoModelForMaskedLM.from_pretrained(modelnames[modelline])
        mtoken="[MASK]"
    elif(modeltypes[modelline]=="xlnet"):
        model = AutoModelForCausalLM.from_pretrained(modelnames[modelline])
        mtoken="<mask>"
    elif(modeltypes[modelline]=="roberta"):
        model = AutoModelForMaskedLM.from_pretrained(modelnames[modelline])
        mtoken="<mask>"
    else:
        raise Exception("Only bert and xlnet are supported, you have entered",modeltypes[modelline])

    # Put Model to cuda, if available
    cudaflag = torch.cuda.is_available()

    if (cudaflag):
        model.to(cudastring)
        
    ###############################
    
    # For each model, initialize Taskloop
    for taskline in range(len(numbofnegs)):
        if(morf[taskline] =="m"):
            pronoun="he"
        elif (morf[taskline]=="f"):
            pronoun="she"
        else:
            raise Exception("Only male and female are supported, you have entered",morf[taskline])
            
        # Somewhat redundant, as probrank is hardcoded now - but too complex to remove
        for rank in toporbottoms:
            toporbottom=rank
            
            
            # Specify outputfilename PROBRUNS  - IMPORTANT: Set manually 
            
            outputfilename="Output/"+printmodelnames[modelline].strip("\n")+"_"+str(toporbottom)
            outputfilename=outputfilename+"_"+morf[taskline]+"_"+numbofnegs[taskline] 
            outputfilename=outputfilename+"_"+negtypes[taskline]+"_"+conjunctions[taskline]
            outputfilename=outputfilename+"_"+additionalels[taskline]+".txt"
            theoutputfile=open(outputfilename,"w")

#            print("\nCurrtestfile:",outputfilename, file=sys.stderr) #DEBUG

            # Initialize variables measuring exwrong

            total_exactlywrongcounter=0
            total_exwrongweights=0
    
            linecounter =0
            start = timer()
            total_exwrongweights=0
            total_exactlywrongcounter=0
            namearray=[]
            
            
            # Make gender adaptations
            
            if(morf[taskline]=="f"):
                namearray=fnamearray
            elif(morf[taskline]=="m"):
                namearray=mnamearray
            else:
                raise Exception("Wrong gender entry, must be m/f")

            # Initialize Name loop 
            
            for currname in namearray:
                difficultenough=0
                if linecounter>10000: # For development
                    break
                strcurrname=str(currname).strip("[]'")
                
                # Initialize PROF loop
                
                for prof in professionsarray:
                    if linecounter>10000: # For development
                        break
        
                    strprof=str(prof).strip("[]'")
                    currsent=strcurrname+" is "+strprof+" and "+pronoun+" likes to "+mtoken+"." # Changed for gerund
#                    currsent=strcurrname+" is "+strprof+" and "+pronoun+" enjoys "+mtoken+"."                
                    # Extract top3token with function
                    top3token, difficultenough=extract_top3token(currsent,linecounter,tokenizer,model)
        
#            print("Top3token:",top3token,"\n",file=sys.stderr) # For debugging
                    toptoken1=str(top3token[0]).strip("[]'")
                    toptoken2=str(top3token[1]).strip("[]'")
                    toptoken3=str(top3token[2]).strip("[]'")
                    linecounter+=1 # Linecounter only increased if prob > threshold
        
                    # Build processable testsentence from template & register number of activities
                    currtestsentence=schemata[taskline].replace('""','"')
#                    print("Looking for act tokens hiere:",currtestsentence,file=theoutputfile)
                    numbacts=1
                    if (re.search("ACT3", currtestsentence)):
#                        print("\tAct 3 token here:",currtestsentence,file=theoutputfile)
                        numbacts=3
                    elif (re.search("ACT2", currtestsentence)):
                        numbacts=2
                    
                   
                    currtestsentence=currtestsentence.replace('FNAME',strcurrname)
                    currtestsentence=currtestsentence.replace('MNAME',strcurrname)
                    currtestsentence=currtestsentence.replace('PROF',strprof)
                    currtestsentence=currtestsentence.replace('MASK',mtoken)
                    currtestsentence=currtestsentence.replace('ACT1',toptoken1)
                    currtestsentence=currtestsentence.replace('ACT2',toptoken2)
                    currtestsentence=currtestsentence.replace('ACT3',toptoken3)
                    currtestsentence=currtestsentence.replace('ACT',toptoken1)
#                    print("Currtestsentence:",currtestsentence, file=sys.stderr) #DEBUG

                    # Initialize Line-wise exwrongweights (for probabilites) and counters (for absolute numbers)
                    exwrongweights=0
                    exactlywrongcounter=0
                    exwrongweights, exactlywrongcounter =predict_check_excluded(currtestsentence, tokenizer,model,modelnames[modelline], top3token, numbacts)
                    total_exwrongweights+=exwrongweights
                    total_exactlywrongcounter+=exactlywrongcounter
                    

            duration = timer() - start
            
#**********************************************************************

#     Print out Statistics
                                                               
            
            print(numbofnegs[taskline],";",negtypes[taskline],";",conjunctions[taskline],end =";",  file=themasteroutputfile)
            print(additionalels[taskline],";",morf[taskline],";",rank,";",modelnames[modelline],end =";",  file=themasteroutputfile)          
            print(total_exwrongweights,";",total_exwrongweights/linecounter,";",total_exactlywrongcounter,";",total_exactlywrongcounter/linecounter,";",linecounter,";",duration,";",schemata[taskline],  file=themasteroutputfile)
                                
                   
    
    
            
            print("\n\nStats:", file=sys.stderr)
            print("\n\nStats:",file=theoutputfile)


            print("Lines processed:",linecounter, file=sys.stderr)
            print("Lines processed:",linecounter,file=theoutputfile)


            print("Computation Time:",duration, file=sys.stderr)
            print("Computation Time:",duration,file=theoutputfile)
            
            print("Exwrong ProbMass (abs/rel):",total_exwrongweights," / ",total_exwrongweights/linecounter,file=sys.stderr)
            print("Exwrong toptoken (abs/rel):",total_exactlywrongcounter,"and",total_exactlywrongcounter/linecounter, file=sys.stderr)
            
            print("Exwrong ProbMass (abs/rel):",total_exwrongweights," / ",total_exwrongweights/linecounter,file=theoutputfile)
            print("Exwrong toptoken (abs/rel):",total_exactlywrongcounter,"and",total_exactlywrongcounter/linecounter, file=theoutputfile)


           
            print("\nModel used:",modelnames[modelline],"Schema used",schemata[taskline], file=sys.stderr)
            print("\nModel used:",modelnames[modelline],"Schema used",schemata[taskline],file=theoutputfile)
            theoutputfile.close()
            
 
##########################################

## Finally: Close files




themasteroutputfile.close()
themnamefile.close()
thefnamefile.close()
professionsfile.close()
