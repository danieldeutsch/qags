dat_dir=$1
dict_file=$2

tok_dir="${dat_dir}/tokenized"
out_dir="${dat_dir}/processed"
mkdir -p ${tok_dir}
mkdir -p ${out_dir}

# tokenize
python fairseq/prepreprocess.py --bert-version bert-base-uncased --data-dir ${dat_dir} --out-dir ${tok_dir}

# preprocess: index, binarize
python fairseq/preprocess.py --source-lang src --target-lang trg \
                             --testpref ${tok_dir}/test.tok \
                             --destdir ${out_dir} --thresholdtgt 10 --thresholdsrc 10 \
                             --srcdict ${dict_file} --tgtdict ${dict_file} \
                             --padding-factor 1 --workers 48;
