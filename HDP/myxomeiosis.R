#myxomeiosis
#Code to simulate SNP data for Victor to play around with HDP models

#model of meiosis: crossovers can occur only at certain sites, and whenever a crossover can occur it does with 50% prob

#simulation model: splits should occur ~every 1% of the chromosome (so expected distance between split points should be 2/3*0.01*chr_length)

rm(list=ls())

#### Meiosis simulation ####
#constants
chr_len = 10000;
split_prob = 1/(2/3*0.01*chr_len); #probability of any particular site being a split site; computed from E[split len]=1/p=2/3*0.01*chr_length
split_pts = which(rbinom(chr_len, 1, split_prob)==1) #crossover points

make.myxomeiosis <- function(split_pts, chr_len){
  myxomeiosis <- function(chr1,chr2){
    #takes 2 chromosomes and produces a new one via pseudo-meiosis where the crossovers happen randomly at predefined points

    #decides where the crossovers occur. Incl 1 as a crossover point so starting parent is randomized.
    splits = split_pts[rbinom(length(split_pts),1,1/2)==1];
    
    #randomize starting person
    if (rbinom(1,1,1/2)==1)
      splits=c(1,splits)

    num_splits = length(splits);
    
    new_chr=chr1;
    
    for (i in 1+2*0:(floor((num_splits)/2)-1)){
      new_chr[splits[i] : (splits[(i+1)]-1)]=chr2[splits[i] : (splits[(i+1)]-1)]
    }  
    
    #for an odd number of splits the last split point to the end should be the swap
    if (num_splits %% 2 == 1) 
      new_chr[splits[num_splits] : chr_len]=chr2[splits[num_splits] : chr_len]
    
    return(new_chr)
  }
  return(myxomeiosis)
}

myx <- make.myxomeiosis(split_pts,chr_len) #wrapped in a closure

#### population simulation ####

make.new_generation <- function(myx) {
  
  new_person <- function (parents,fam_id=1){
    #takes a list (father,mother) and returns a child
    #fam_id is can be used to track common parents
    #the sex of the parent doesn't actually matter
    chr1=myx(parents[[1]]$chr$chr1,parents[[1]]$chr$chr2);
    chr2=myx(parents[[2]]$chr$chr1,parents[[2]]$chr$chr2);
    new_chr <- data.frame(chr1,chr2,stringsAsFactors=FALSE)

    return(list(chr=new_chr,sib_id=fam_id,par_sib_id=c(parents[[1]]$sib_id,parents[[2]]$sib_id)))
  }
  
  new_generation <- function (breeders,num_kids=2,fam_id=1){
    #takes a list of members of a previous generation, pairs them off randomly and produces num_kids from each
    #first cousins and closer are forbidden from breeding
    #sib id tracks the sibling information of the current iteration (each function call is a new family)
    
    #quit if less than 2 breeders
    if (length(breeders)<2) return (list())
    
    #pick the first parent
    father <- sample(length(breeders),1);
    
    #extract the parent family information
    #this can't be the best way to do this
    par_sib_id=rbind(rep(0,length(breeders)),rep(0,length(breeders)))
    for (i in 1:length(breeders)){
      par_sib_id[,i]=breeders[][[i]]$par_sib_id
    }
    
    #the people whose parents are not siblings of the father's parents
    #todo: inspection of generated data suggests this isn't completely preventing siblings from mating; why not?
    unrelated<-intersect(
      intersect(which(par_sib_id[1,]!=breeders[[father]]$par_sib_id[1]),which(par_sib_id[1,]!=breeders[[father]]$par_sib_id[2])),
      intersect(which(par_sib_id[2,]!=breeders[[father]]$par_sib_id[1]),which(par_sib_id[2,]!=breeders[[father]]$par_sib_id[2])))                   
    
    #quit when there are no longer 2 unrelated people
    if (length(unrelated)==0) return (list())
    
    #pick an unrelated mother to reproduce
    mother <- sample(unrelated,1);
    
    prnts=c(father,mother);
    
    kids=rep(list(breeders[1]),num_kids) #preallocate
    for (i in 1:num_kids){
      kids[[i]]=new_person(breeders[prnts],fam_id)
    }
    
    return (append(new_generation(breeders[-prnts],num_kids,(fam_id+1)),kids))
  }
  
  return(new_generation)
}

new_gen <- make.new_generation(myx)

##actually generating data now
#a person is 2 chromosomes, a sibling id and a pair of parental sibling ids
#the last two features are included as an incest prevention mechanism, since this screws up sims
#mix
num_generations=25;
num_kids=2;

###first lets play around with just 26 founders and see how mixing works
num_founders=26;
#this is for testing, it assigns each chromosome site to be "(founder_id)(site location)"
gen_chrom <- function (letter){
  chr1=rep(paste(letter,1,sep=""),chr_len)
  chr2=rep(paste(letter,2,sep=""),chr_len)
  return(data.frame(chr1,chr2,stringsAsFactors=FALSE))
}
founder_chroms = apply(t(as.vector(letters)),2,gen_chrom) #each column is the chromosome of a founder
founder_parent_ids = rbind(1:num_founders,num_founders+1:2*num_founders) #unique parents
founder_sib_ids = 1:num_founders #unique sibling ids


###now for a proper simulation
num_founders=150;
source_chrom <- rbeta(chr_len,2,2); #the typical distribution for founder chromosomes
gen_chrom <- function (letter){
  chr1=rbinom(chr_len,size=1,prob=source_chrom)
  chr2=rbinom(chr_len,size=1,prob=source_chrom)
  return(data.frame(chr1,chr2,stringsAsFactors=FALSE))
}
founder_chroms = apply(t(1:num_founders),2,gen_chrom) #each column is the chromosome of a founder
founder_parent_ids = rbind(1:num_founders,num_founders+1:2*num_founders) #unique parents
founder_sib_ids = 1:num_founders #unique sibling ids



founders <- list(list(chr=founder_chroms[[num_founders]],sib_id=founder_sib_ids[num_founders],par_sib_id=founder_parent_ids[,num_founders]))
for (i in (num_founders-1):1){
  founders <- append(list(list(chr=founder_chroms[[i]],sib_id=founder_sib_ids[i],par_sib_id=founder_parent_ids[,i])),
                     founders)
}

worst_generation <- new_gen(founders,2)
for (i in 1:num_generations){
  #try to keep it around twice the number of founders; this is probably unrealistically small
  if (length(worst_generation) < num_founders) worst_generation <- new_gen(worst_generation,4) else
    worst_generation <- new_gen(worst_generation,2) 
}


###simulate observed data###
#observed data actually just tells us whether, for each site, there's 00, 11 or mixed (or missing)

make.obs_data_extract <- function(){
  obs_data_extract <- function(person){
    obs_data=rep(2,chr_len); #2 codes a mismatch
    obs_data[person$chr$chr1 == person$chr$chr2]=person$chr$chr1[person$chr$chr1 == person$chr$chr2];
    return(obs_data)
  }
  return(obs_data_extract)
}

get_obs_data <- make.obs_data_extract()

#this is a list of observed data for each person
obs_data <- lapply(worst_generation, get_obs_data)

###some further helper functions

unique_haplotypes <- function(generation,loci=(1:chr_len))
#this eats a generation and determines the number of unique haplotypes at sites [loci]
{
  person_chroms <- rbind(generation[[1]]$chr$chr1[loci],generation[[1]]$chr$chr2[loci])
  if (length(generation)==1) return (unique(person_chroms)) else
  return (unique(rbind(person_chroms,unique_haplotypes(generation[-1],loci))))
}