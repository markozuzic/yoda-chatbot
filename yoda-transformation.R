library(coreNLP)
library(rJava)
library(readr)
library(stringi)
library(tokenizers)
library(stringr)

#downloadCoreNLP()
initCoreNLP(type = "english_all")

train_dec <- read_delim("C:/Users/M/Desktop/siap_data/train.dec", 
                         "\n", escape_double = FALSE, col_names = FALSE, 
                         trim_ws = TRUE)

clean_data <- function(x) {
  x <- iconv(x, "UTF-8", "UTF-8",sub='')
  x <- tolower(x)
  x <- gsub("...", "", x, fixed = TRUE)
  x <- gsub("<i>", "", x, fixed = TRUE)
  x <- gsub("</i>", "", x, fixed = TRUE)
  x <- gsub("<b>", "", x, fixed = TRUE)
  x <- gsub("<u>", "", x, fixed = TRUE)
  x <- gsub("</b>", "", x, fixed = TRUE)
  x <- gsub("</u>", "", x, fixed = TRUE)
  x <- gsub("*", "", x, fixed = TRUE)
  x <- gsub("--", "", x, fixed = TRUE)
  x <- gsub("\\'ll", " will", x)
  x <- gsub("\\'m", " am", x)
  x <- gsub("(^he|^she|^it|^that|^what)\\'s", "\\1 is", x)
  x <- gsub("\\'re", " are", x)
  x <- gsub("won\\'t", "will not", x)
  x <- gsub("can\\'t", "cannot", x)
  x <- gsub("n\\'t", " not", x)
  x <- gsub("n\\'", "ng", x)
  x <- gsub("-+", "", x)
  x <- gsub(",", "", x)
  x <- gsub("\\\"", "", x)
  #in the proces of removing the lines multiple whitespaces will be left
  x <- gsub(" +", " ", x)
  return(x)
}

punc_space <- function(x) {
  x <- gsub("?", " ?", x, fixed = TRUE)
  x <- gsub("!", " !", x, fixed = TRUE)
  x <- gsub(".", " .", x, fixed = TRUE)
  return(x)
}

find_dependent <- function(list, annotation, s=c()) {
  if(identical(list, character(0))) {
    return(s)
  }
  for (l in list) {
    nested <-
      c(annotation$basicDep$dependentIdx[annotation$basicDep$governorIdx == l &
                                          annotation$basicDep$type != "punct"])
    s <- c(s, nested)
  }
  find_dependent(nested, annotation, s)
}

find_and_sort <- function(list, annotation) {
  d <- find_dependent(list, annotation)
  d <- unique(c(list, d))
  d <- sort(d)
  return(d)
}

convert_idx <- function(idx_list, annotation) {
  words <- vector()
  for(i in idx_list) {
    words <- c(words, annotation$basicDep$dependent[annotation$basicDep$dependentIdx == i])
  }
  words <- paste(words, collapse = " ")
  return(words)
}

check_predicate_type <- function(idx_list, root_idx, annotation) {
  for(i in idx_list) {
    if(annotation$basicDep$type[annotation$basicDep$dependentIdx == i] %in% 
       c("advmod", "conj", "cc")) {
      root_idx <- c(root_idx, i)
    }
  }
  return(sort(unique(root_idx)))
}

yoda_transform <- function(sen) {
  
  sen <- clean_data(sen)
  
  if(is.na(sen)) {
    transformed <<- c(transformed, FALSE)
    return(sen)
  }
  
  if(str_count(sen, "\\S+") <= 2) {
    transformed <<- c(transformed, TRUE)
    return(sen)
  }

  annotation <- annotateString(sen)

  subj_idx <- annotation$basicDep$dependentIdx[grepl("subj", annotation$basicDep$type)]
  if(length(subj_idx) > 1 || length(subj_idx) == 0) {
    transformed <<- c(transformed, NA)
    return(sen)
  }
  
  punct_idx <- annotation$basicDep$dependentIdx[annotation$basicDep$type == "punct"]
  
  subj_idx_list <- find_and_sort(subj_idx, annotation)
  subject <- convert_idx(subj_idx_list, annotation)
  

  root_idx <- annotation$basicDep$dependentIdx[annotation$basicDep$type == "root"]

  obj_idx <- annotation$basicDep$dependentIdx[grepl("obj", annotation$basicDep$type)]

  if(!identical(obj_idx, character(0))) { #if the object_idx isn't char(0), we found the object
    obj_idx_list <- find_and_sort(obj_idx, annotation)
    object <- convert_idx(obj_idx_list, annotation)
  
    pred_idx_list <- find_and_sort(root_idx, annotation)
    pred_idx_list <- pred_idx_list[!(pred_idx_list %in% c(subj_idx_list, obj_idx_list, punct_idx))]
    pred_idx_list <- as.character(sort(as.numeric(pred_idx_list)))
    predicate <- convert_idx(pred_idx_list, annotation)
  } else {
  #if we have an aux or cop, root is the object
    if(sum(grepl("aux", annotation$basicDep$type)) > 0 |
       sum(grepl("cop", annotation$basicDep$type)) > 0) { 
      pred_idx <- annotation$basicDep$dependentIdx[grepl("aux", annotation$basicDep$type)]
      if(identical(pred_idx, character(0))) {
        pred_idx <- annotation$basicDep$dependentIdx[grepl("cop", annotation$basicDep$type)]
      }
      pred_idx_list <- find_and_sort(pred_idx, annotation)
      #pred_idx_list <- as.character(sort(as.numeric(pred_idx_list)))
      predicate <- convert_idx(pred_idx_list, annotation)
    
      #root is the object
      obj_idx_list <- find_and_sort(root_idx, annotation)
      #we have to exclude idx that are in the subject or predicate list
      obj_idx_list <- obj_idx_list[!(obj_idx_list %in% c(subj_idx_list, pred_idx_list, punct_idx))]
      obj_idx_list <- as.character(sort(as.numeric(obj_idx_list)))
      object <- convert_idx(obj_idx_list, annotation)
    }else {
      #in the case where there isn't an aux or cop, root is the predicate, 
      #advmod, cc and
      root_idx_list <- annotation$basicDep$dependentIdx[annotation$basicDep$governorIdx == root_idx]
      root_idx_list <- c(root_idx, root_idx_list)
      root_idx_list <- root_idx_list[!(root_idx_list %in% c(subj_idx_list, punct_idx))]
      pred_idx_list <- check_predicate_type(root_idx_list, root_idx, annotation)
      predicate <- convert_idx(pred_idx_list, annotation)
    
      obj_idx <- root_idx_list[!(root_idx_list %in% c(subj_idx_list, pred_idx_list, punct_idx))]
      obj_idx_list <- find_and_sort(obj_idx, annotation)
      obj_idx_list <- obj_idx_list[!(obj_idx_list %in% c(subj_idx_list, pred_idx_list, punct_idx))]
      obj_idx_list <- as.character(sort(as.numeric(obj_idx_list)))
      
      object <- convert_idx(obj_idx_list, annotation)
    }
  }
  
  punct <- convert_idx(punct_idx, annotation)

  sentence <- paste(object, subject, predicate, collapse = " ")
  sentence <- trimws(paste(sentence, punct, sep = ""))
  transformed <<- c(transformed, TRUE)
  return(sentence)
}

yoda <- function(sentence) {
  sentence <- iconv(sentence, "UTF-8", "UTF-8",sub='')
  sentence <- gsub("...", "", sentence, fixed = TRUE)
  sentence <- gsub("<i></i>", "", sentence, fixed = TRUE)
  sentence <- gsub("--", "", sentence, fixed = TRUE)
  sentence <- gsub(" .", ".", sentence, fixed = TRUE)
  
  single_sentences <- unlist(tokenize_sentences(sentence))
  yoda_sentence <- vector()
  for(i in seq(1,length(single_sentences))) { 
    yoda_sentence <- trimws(paste(yoda_sentence, yoda_transform(single_sentences[i])))
  }
  transformed_global <<- c(transformed_global, ifelse(sum(is.na(transformed)) > 0, FALSE, TRUE))
  #restart the counter
  transformed <<- vector()
  #yoda_sentence <- clean_data(yoda_sentence)
  return(yoda_sentence)
}

transform <- function(x) {
  #dft <- data.frame()
  j <- 1
  for(i in x){
    counter <<- counter + 1
    print(counter)
    dft[j,1] <<- try(yoda(i), transformed_global <<- c(transformed_global, FALSE))
    j <- j + 1
  }
  return(dft)
}

transformed_global <- vector() 
transformed <- vector()
counter <- 0
dft <- data.frame()

df <- lapply(train_dec[48001:107852,], transform)

#subject <- 1st nsujb, predicate <- root, object <- the rest

