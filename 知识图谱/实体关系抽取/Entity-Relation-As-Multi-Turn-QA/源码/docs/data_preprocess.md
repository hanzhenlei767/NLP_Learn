# Data Preprocess Guideline 


We release preprocessed and source datasets including `ACE2004`, `ACE2005`, `CoNLL2004`. 

1. Download original datasets from:
    * ACE 2004 `https://catalog.ldc.upenn.edu/LDC2005T09`
    * ACE 2005 `https://catalog.ldc.upenn.edu/LDC2006T06`
    * CoNLL 2004 `https://cogcomp.seas.upenn.edu/Data/ER/conll04.corp`
2. Split datasets following the previous work: 
    * ACE 2004 `https://github.com/tticoin/LSTM-ER/`
    * ACE 2005 `https://github.com/tticoin/LSTM-ER/`
    * CoNLL 2004 `https://github.com/bekou/multihead_joint_entity_relation_extraction/tree/master/data/CoNLL04`
3. Transform data to Question-Answering scheme:
    
    Convert data from `Relation(Entity1, Entity2)` to `(Question, Answer, Context)`. 
    
    Take `ACE 2004` dataset for example: 

    ```bash 
    export TASK_NAME=ace2004 
    export ORIGIN_DATA_PATH=/path/to/ace2004
    export EXPORT_DATA_PATH=/path/to/convert_qa_scheme
    
    cd utils/
    python3 prep_qa_data.py --data_sign $TASK_NAME \
        --origin_data_path $ORIGIN_DATA_PATH \
        --export_data_path $EXPORT_DATA_PATH 
    ```
    After this, you will have a `ace2004` subdirectory under the folder of `/path/to/convert_qa_scheme`. The folder of `ace2004` contains the experiments files for entity and relation extraction tasks. 
    
    The `TASK_NAME` can be `ace2004`, `ace2005`, `conll2004`.
    
    The folder of `ace2004` contains train/validate/test files for the task of entity extraction and relation classification. 