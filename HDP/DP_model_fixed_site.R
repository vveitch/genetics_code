###Haplotype fitter
#the idea here is to determine the haplotypes at a fixed set of loci
#we'll start off with no mutation and no measurement error
#so the only randomness is the ambiguity at mixed sites

#this is mostly as a Dirichlet Process warmup exercise for myself

rm(list=ls())

#constants
#ground truth for loci 1:7 is 44 haplotypes (see helper function in
#myxomeiosis.R)
kLoci=1:7; 
kNumLoci=length(kLoci);

###Extract data for the loci we care about in a relevant form
## outputs: 
#   obs: a list of the allele strings observed for each individual at the loci
#   gene_labels: the list of alleles and their counts that can be
#   deterministically inferred
data_extract <- function(){
  load("~/Code/Genetics/Data/150_found_25_gen_observed_data.rdata")
  
  n=length(obs_data);
  chr_len=length(obs_data[[1]]);
  relevant_data <- matrix(data=1,nrow=kNumLoci,ncol=n)
  for (i in (1:n)){
    relevant_data[,i]=obs_data[[i]][kLoci]
  } 
  
  #the next step is to assign initial haplotype labels to all the observations
  rand_haplo<-function(sing_obs){
    mix_indices = which(sing_obs==2);
    lbl_choice = rbinom(length(mix_indices),1,1/2);
    father = sing_obs; father[mix_indices]=lbl_choice;
    mother = sing_obs; mother[mix_indices]=!lbl_choice;
    return (list(father=father,mother=mother))
  }
  
  #assign random chromosomes to each observation and convert to string for ease
  #of later comparsion
  data_labels <- apply(relevant_data,2,rand_haplo);
  
  #some stuff to get out initial haplotype counts, from the deterministic
  #observations and the random initial assignments
  init_table<-table(unlist(lapply(data_labels,function(x) lapply(x,paste,collapse=","))));
  init_labels<-lapply(names(init_table),FUN=(function(x) as.integer(unlist(strsplit(x,split=",")))))
  names(init_labels)<-names(init_table); #include the *strings* to make later searching easier
  haplo_clusters=list(word=init_labels,count_table=init_table)
  
  #data with no mixed components can be treated deterministically, so we filter
  #this out now
  is.deterministic <- function(observation_snippet){
    if (all(observation_snippet!=2)) return(TRUE)
    else return (FALSE)
  }
  
  det_indices<-apply(X=relevant_data,MARGIN=2,FUN=is.deterministic)
  #this is the data needs to be labelled
  observations <- list(combo=relevant_data[,!det_indices],lbls=data_labels[!det_indices]);
  
  
  return (list(obs=observations,init_haplo_clusters=haplo_clusters))
}

#each "cluster" is a defining word and the number already assigned to that word
tmp<-data_extract();
obs<-tmp$obs; haplo_clusters<-tmp$init_haplo_clusters;
rm(tmp); rm(data_extract);


###DP Gibbs sampler 
#the idea is to sample from the posterior distribution of the labels at each
#site rough idea: put a prior on haplotypes equiv to DP with base measure
#categorical distribution over all (2^|{loci}|) possible words 
#this may not really make much sense in this context (since why not use a
#regular Dirichlet distribution?), but it generalizes easily to more complicated
#scenarios
#each observation gets two labels, the haplotype on each chromo.
#p(x|k_1,k_2)=0 or 2^(-m), where m is the number of "2"s in x (ie. labels are
#all either impossible or equiprobable)
#p(k_1,k_2)=p(k_2|k_1)p(k_1) (so if a new cluster needs to be created then...)

#remark: the exchangeability of the dirichlet prior and the llhd is critical for
#our setup (it's what allows us to treat the labels as *unordered* pairs)

kNumObs=dim(obs$combo)[2];

llhd <- function(observation,haplo_clusters){
  #this is to compute the likelihood p(x|k_1,k_2) for all *unordered* pairs of labels
  #ARGS: 
  #observation: a vector of allele observations at the loci
  #haplo_clusters: a list of currently instantianted labels and the associated counts
  
  #under our model p(x|k_1,k_2) for already instantiated k_1,k_2 is either 0 or 
  #constant, so llhd just checks for compatibility figure out which currently 
  #instantiated clusters are already compatible
  deterministic_indices<-(observation!=2)
  compatible<-which(unlist(lapply(haplo_clusters$word[],function(x) all(x[deterministic_indices]==observation[deterministic_indices]))))
  
  pairup <- function(compat_indices=compatible){
    #recursively match up the already instantiated compatible indices each
    #component of the output indicates the pair of labels and how many (0 or 1)
    #are newly instantiated
    #The llhds are determined entirely by whether both clusters are already
    #instantiated (for new clusters the llhd has a factor accounting for the
    #probability that newly generated thing will be the unique compatible
    #string)    
    
    if (length(compat_indices)==0) return (list(lbls=list(),clus_llhd=vector(),haplo_counts=vector()))      
    
    father<-haplo_clusters$word[compat_indices][[1]]
    #there is only one mother compatible with data and father
    #this will need to be changed for smarter implementations
    mother<-father; mother[!deterministic_indices]<-!father[!deterministic_indices]
    
    #now we need some logic depending if the mother haplotype is an already instantiated cluster
    mother_index <- match(paste(mother,collapse=","),names(haplo_clusters$word[compat_indices]),nomatch=0)
    
    if (mother_index==0){
      #append is very slow, but the lists I'm dealing with should generally be
      #quite small... I hope. I wish R had natural push functionality
      recurs_pairs <-pairup(compat_indices[-1])
      return (list(lbls=append(list(list(father=father,mother=mother)),recurs_pairs$lbls),
                   clus_llhd=c((1/2^kNumLoci),recurs_pairs$clus_llhd),
                   haplo_counts=cbind(rbind(as.integer(haplo_clusters$count_table[1]),0),
                                      recurs_pairs$haplo_counts)))
    } else {
      recurs_pairs <-pairup(compat_indices[-c(1,mother_index)])
      return (list(lbls=append(list(list(father=father,mother=mother)),recurs_pairs$lbls),
                   clus_llhd=c(1,recurs_pairs$clus_llhd),
                   haplo_counts=cbind(rbind(as.integer(haplo_clusters$count_table[1]),as.integer(haplo_clusters$count_table[mother_index])),
                                      recurs_pairs$haplo_counts)))
    }
  }
  
  pairs<-pairup();
  
  #there's one more option, which is that both labels are new 
  #we will generate the corresponding new thing only later in the case that it's
  #selected (which is very low probability)
  return(list(lbls=append(list(list(father=rep(-1,kNumLoci),mother=rep(-1,kNumLoci))),pairs$lbls),
              clus_llhd=c((1/2^kNumLoci),pairs$clus_llhd),
              haplo_counts=cbind(rbind(0,0),
                                 pairs$haplo_counts)))
}


#dirichlet prior for already instantianted kth cluster is N_k/(N+alpha) and
#alpha/(N+alpha) for new cluster

dp_prior<-function(cluster_counts,alpha){
  #gives (up to normalization) the DP prior
  #ARGS: 
  # cluster counts: the counts of the number of data points assigned to each label at the present moment
  # alpha: the DP concentration parameter
  
  #note that the 2 clusters can never be the same
  if (cluster_counts[1] == 0 && cluster_counts[2] == 0) {return (alpha^2)    
  } else if (cluster_counts[1] == 0) {return (alpha*cluster_counts[2])
  } else if (cluster_counts[2] == 0) {return (alpha*cluster_counts[1])
  } else {return (cluster_counts[1]*cluster_counts[2]) }
}

###now we finally get to the sampler

#I'm setting this in a pretty cheaty way for testing, using that the number of
#clusters is asymptotically alpha*log(n) and the true value is 44
kAlpha=17; 
kNumIter=500; #way too low, just for simple testing

for (i in 1:kNumIter){
  #randomly permute the data at each step
  for (j in sample(1:kNumObs,size=kNumObs,replace=F)){
    #remove the labels of the jth data point from the counts
    father_index <- match(paste(obs$lbls[[j]]$father,collapse=","),names(haplo_clusters$count_table),nomatch=0)
    if (haplo_clusters$count_table[father_index]==1) {
      haplo_clusters$count_table<-haplo_clusters$count_table[-(father_index)]
      haplo_clusters$word<-haplo_clusters$word[-(father_index)]
    } else {
      haplo_clusters$count_table[father_index]<- haplo_clusters$count_table[father_index]-1
    }
    
    
    mother_index <- match(paste(obs$lbls[[j]]$mother,collapse=","),names(haplo_clusters$count_table),nomatch=0)
    if (haplo_clusters$count_table[father_index]==1) {
      haplo_clusters$count_table<-haplo_clusters$count_table[-(mother_index)]
      haplo_clusters$word<-haplo_clusters$word[-(mother_index)]
    } else {
      haplo_clusters$count_table[mother_index]<- haplo_clusters$count_table[mother_index]-1
    }
    
    #sample from the posterior distribution
    llhd_stuff <- llhd(obs$combo[,j],haplo_clusters)
    unnorm_post<-(apply(llhd_stuff$haplo_counts,2,dp_prior,alpha=kAlpha))*llhd_stuff$clus_llhd
    draw_index <- match(1,rmultinom(n=1,size=1,prob=unnorm_post))
    
    #now we need some logic depending on whether 1,2 or no new clusters were
    #instantiated. This updates cluster counts to reflect the new assignment.
    if (draw_index==1){
      #2 new clusters are created, so we need to draw from the base distribution
      mix_indices = which(obs$combo[,j]==2);
      lbl_choice = rbinom(length(mix_indices),1,1/2);
      father = obs$combo[,j]; father[mix_indices]=lbl_choice;
      mother = obs$combo[,j]; mother[mix_indices]=!lbl_choice;
      
      #because we are drawing from a base distribution with finite support we might have collisions
      father_index <- match(paste(father,collapse=","),names(haplo_clusters$count_table),nomatch=0)
      if (father_index==0){
        #no collisions, we need to create the 2 new cluster labels (if it was
        #only 1 that needed to be created then the draw index wouldn't be 1)
        father_name<-paste(father,collapse=",")
        mother_name<-paste(mother,collapse=",")
        num_clust<-length(haplo_clusters$count_table);
        
        haplo_clusters$count_table <- c(haplo_clusters$count_table,1,1)
        names(haplo_clusters$count_table)[c(num_clust,(num_clust+1))]=
          c(father_name,mother_name)
        
        haplo_clusters$word=append(haplo_clusters$word,list(father,mother))
        names(haplo_clusters$word)[c(num_clust,(num_clust+1))]=
          c(father_name,mother_name)
      } else {
        #if there was a collision we just need to update the counts as appropriate
        #is it gaurenteed that if the father collides then the mother does as well?
      }
      
    } else if (llhd_stuff$haplo_counts[draw_index]==0){
      
    } else {
      
    }
  }
}
