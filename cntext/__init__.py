__version__ = "2.2.1"


from .hello import hello, welcome
from .stats.index import fepu, epu, semantic_brand_score, word_hhi
from .stats.readability import readability
from .stats.sentiment import sentiment, sentiment_by_valence
from .stats.similarity import cosine_sim, jaccard_sim, minedit_sim, simple_sim
from .stats.utils import word_count, word_in_context

from .io.mda import extract_mda
from .io.file import read_pdf, read_file, read_docx, read_files, get_files
from .io.dict import build_yaml_dict, read_yaml_dict, get_dict_list
from .io.utils import get_cntext_path, detect_encoding, fix_text, fix_contractions, traditional2simple, clean_text

from .model.glove import GloVe
from .model.w2v import Word2Vec
from .model.fasttext import FastText
from .model.sopmi import SoPmi
from .model.utils import expand_dictionary, co_occurrence_matrix, load_w2v, glove2word2vec, evaluate_similarity, evaluate_analogy
from .mind import procrustes_align,divergent_association_task, sematic_distance, project_word, project_text,sematic_projection, semantic_centroid, discursive_diversity_score, generate_concept_axis, wepa
from .plot import lexical_dispersion_plot1, lexical_dispersion_plot2, matplotlib_chinese

from .llm import analysis_by_llm,text_analysis_by_llm, llm
#from .psymatrix import PsyMatrixCalculator, CONCEPTS_PAIRS, plot_semantic_difference