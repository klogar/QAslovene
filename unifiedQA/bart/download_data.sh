#!/bin/sh

mkdir -p data

BASE_URL=https://storage.googleapis.com/danielk-files/data

# datasets for pretraining
for dataset in narrativeqa squad2 boolq ; do
    mkdir -p data/${dataset}
    for data_type in train dev test ; do
        wget ${BASE_URL}/${dataset}/${data_type}.tsv -O data/${dataset}/${data_type}.tsv
    done
done

# other datasets
#for dataset in qasc qasc_with_ir commonseseqa openbookqa_with_ir arc_hard_with_ir arc_easy_with_ir winogrande_xl physical_iqa social_iqa ropes natural_questions_with_dpr_para ; do
#    mkdir -p data/${dataset}
#    for data_type in train dev test ; do
#        wget ${BASE_URL}/${dataset}/${data_type}.tsv -O data/${dataset}/${data_type}.tsv
#    done
#done


exit

wget https://nlp.cs.washington.edu/ambigqa/data/nqopen.zip -O data/nqopen.zip
unzip -d data data/nqopen.zip
rm data/nqopen.zip

