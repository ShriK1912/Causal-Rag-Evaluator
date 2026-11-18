"""
Synthetic Data and Vector Database Generator for Causal RAG Diagnostician.
Generates:
1. A local persistent ChromaDB collection with 120+ domain documents.
2. A controlled CSV dataset of failed RAG queries covering all 5 failure classes:
   - Retrieval Miss / Database Gap
   - Ranking Error
   - Context Contamination
   - Hallucination
   - Contradiction
"""

import sys
from pathlib import Path
import pandas as pd
import chromadb
from chromadb.config import Settings

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    SYNTHETIC_DATA_DIR,
    FAILURE_CATEGORIES,
)


def build_corpus_documents():
    """Generates a diverse set of academic documents across multiple fields."""
    docs = []
    
    # Domain 1: Machine Learning & AI
    docs.append({"id": "doc_ml_001", "text": "Supervised learning algorithms require labeled datasets to learn mapping functions from inputs to outputs.", "domain": "Machine Learning"})
    docs.append({"id": "doc_ml_002", "text": "Unsupervised learning discovers hidden patterns, clusters, or structures within unlabeled data representations.", "domain": "Machine Learning"})
    docs.append({"id": "doc_ml_003", "text": "Support Vector Machines seek a maximum-margin hyperplane separating classes in high-dimensional feature space.", "domain": "Machine Learning"})
    docs.append({"id": "doc_ml_004", "text": "Random Forests construct an ensemble of decorrelated decision trees using bootstrap aggregation and random feature subsets.", "domain": "Machine Learning"})
    docs.append({"id": "doc_ml_005", "text": "Principal Component Analysis transforms correlated features into orthogonal linear components capturing maximal variance.", "domain": "Machine Learning"})
    docs.append({"id": "doc_ml_006", "text": "K-Means clustering partitions observations into k Voronoi cells by iteratively updating cluster centroids.", "domain": "Machine Learning"})
    docs.append({"id": "doc_ml_007", "text": "DBSCAN clusters data based on spatial density connectivity and isolates sparse observations as noise points.", "domain": "Machine Learning"})
    docs.append({"id": "doc_ml_008", "text": "Gradient Descent minimizes loss functions by iteratively calculating partial derivatives and updating parameter weights.", "domain": "Machine Learning"})
    docs.append({"id": "doc_ml_009", "text": "Overfitting occurs when a statistical model memorizes sample noise rather than learning generalizable underlying trends.", "domain": "Machine Learning"})
    docs.append({"id": "doc_ml_010", "text": "Cross-validation estimates generalization performance by partitioning training samples into k stratified subsets.", "domain": "Machine Learning"})
    docs.append({"id": "doc_ml_011", "text": "Transformer architectures rely on multi-head scaled dot-product self-attention to model long-range sequence dependencies.", "domain": "Machine Learning"})
    docs.append({"id": "doc_ml_012", "text": "Retrieval-Augmented Generation couples external vector indexing with generation models to ground synthesis in source text.", "domain": "Machine Learning"})
    docs.append({"id": "doc_ml_013", "text": "Cosine similarity measures the angle between normalized vector embeddings, invariant to vector magnitude.", "domain": "Machine Learning"})
    docs.append({"id": "doc_ml_014", "text": "Dense passage retrieval uses bi-encoders to map queries and passages into a shared vector space for nearest-neighbor search.", "domain": "Machine Learning"})
    docs.append({"id": "doc_ml_015", "text": "BM25 is a probabilistic term-matching retrieval function based on inverse document frequency and document length normalization.", "domain": "Machine Learning"})
    
    # Specific targeted document for Ranking Error experiment
    docs.append({"id": "doc_target_rank_01", "text": "Project Kepler was officially launched on March 7, 2009 by NASA to discover Earth-sized planets orbiting other stars.", "domain": "Astrophysics"})
    docs.append({"id": "doc_target_rank_02", "text": "The Haber-Bosch chemical process synthesizes ammonia from nitrogen and hydrogen gas under high temperature and iron catalyst.", "domain": "Chemistry"})
    docs.append({"id": "doc_target_rank_03", "text": "Ada Lovelace wrote the first published algorithm intended for implementation on Babbage's mechanical Analytical Engine in 1843.", "domain": "Computer Science History"})
    docs.append({"id": "doc_target_rank_04", "text": "The Crispr-Cas9 endonuclease system enables targeted genomic edits by guiding RNA-directed double-strand DNA cleavage.", "domain": "Genetics"})
    docs.append({"id": "doc_target_rank_05", "text": "Turing completeness describes an abstract computational system capable of simulating any single-taped Turing machine.", "domain": "Theoretical Computer Science"})

    # Specific documents for Contradiction pairs
    docs.append({"id": "doc_contra_a1", "text": "Consensus Report Alpha: The Neanderthal extinction occurred approximately 40,000 years ago primarily due to rapid climatic cooling.", "domain": "Paleoanthropology"})
    docs.append({"id": "doc_contra_b1", "text": "Consensus Report Beta: Neanderthals survived until 28,000 years ago in southern Iberia, outlasting northern populations despite climatic shifts.", "domain": "Paleoanthropology"})
    docs.append({"id": "doc_contra_a2", "text": "Historical Archive X: The printing press was independently invented by Lourens Janszoon Coster in Haarlem in 1439.", "domain": "History"})
    docs.append({"id": "doc_contra_b2", "text": "Historical Archive Y: Johannes Gutenberg invented movable type printing in Mainz Germany around the year 1440.", "domain": "History"})
    docs.append({"id": "doc_contra_a3", "text": "Pharmacology Bulletin 1: Drug compound Z9 reduces hypertension by 45% with zero reported cardiac arrhythmias in clinical trials.", "domain": "Medicine"})
    docs.append({"id": "doc_contra_b3", "text": "Pharmacology Bulletin 2: Drug compound Z9 was terminated in Phase II after inducing severe cardiac arrhythmias in 18% of patients.", "domain": "Medicine"})

    # Specific documents for Context Contamination (distractor topics with misleading shared tokens)
    docs.append({"id": "doc_distract_01", "text": "The Python programming language named after Monty Python was created by Guido van Rossum in the late 1980s.", "domain": "Programming"})
    docs.append({"id": "doc_distract_02", "text": "Python regius, known commonly as the royal ball python, is a non-venomous constrictor snake native to West and Central Africa.", "domain": "Zoology"})
    docs.append({"id": "doc_distract_03", "text": "Apple Inc is an American multinational technology company headquartered in Cupertino, California, producing consumer electronics.", "domain": "Business"})
    docs.append({"id": "doc_distract_04", "text": "The domesticated apple tree Malus domestica produces sweet pomaceous fruit originating from Central Asia wild ancestors.", "domain": "Botany"})
    docs.append({"id": "doc_distract_05", "text": "Java is an island of Indonesia bordered by the Indian Ocean to the south and the Java Sea to the north.", "domain": "Geography"})
    docs.append({"id": "doc_distract_06", "text": "Java technology provides a class-based, object-oriented language developed by James Gosling at Sun Microsystems.", "domain": "Software"})

    # Domain 2: Astrophysics & Space Exploration
    for i in range(1, 26):
        docs.append({
            "id": f"doc_astro_{i:03d}",
            "text": f"Astrophysical observation bulletin {i}: Stellar nucleosynthesis in massive asymptotic giant branch stars produces heavy elements via the s-process neutron capture pathway under temperature condition T={i*10}MK.",
            "domain": "Astrophysics"
        })

    # Domain 3: Molecular Biology & Medicine
    for i in range(1, 26):
        docs.append({
            "id": f"doc_bio_{i:03d}",
            "text": f"Biochemical pathway record {i}: Ribosomal translation initiation factors recruit messenger RNA subunits to facilitate peptide bond elongation during eukaryotic cellular metabolic cycle {i}.",
            "domain": "Molecular Biology"
        })

    # Domain 4: Quantum Physics & Thermodynamics
    for i in range(1, 26):
        docs.append({
            "id": f"doc_quantum_{i:03d}",
            "text": f"Quantum thermodynamics theorem {i}: Non-equilibrium quantum state entanglement across spin chains exhibits topological phase transitions with critical scaling exponent nu={i*0.1:.2f}.",
            "domain": "Quantum Physics"
        })

    # Domain 5: Environmental Science & Climatology
    for i in range(1, 26):
        docs.append({
            "id": f"doc_env_{i:03d}",
            "text": f"Paleoclimatic ice core study {i}: Atmospheric CO2 concentrations trapped in Antarctic glacial ice sheets reveal cyclic orbital Milankovitch periodicity over cycle duration {i*10000} years.",
            "domain": "Climatology"
        })

    # Domain 6: World History & Economics
    for i in range(1, 26):
        docs.append({
            "id": f"doc_hist_{i:03d}",
            "text": f"Economic history archive {i}: Maritime trade routes connecting Mediterranean merchant guilds established monetary credit instruments and bills of exchange in fiscal period {1300 + i*10}.",
            "domain": "Economics"
        })

    return docs


def populate_chromadb():
    """Initializes local persistent ChromaDB and populates it with corpus."""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    
    # Reset or get collection
    try:
        client.delete_collection(name=DEFAULT_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=DEFAULT_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "description": "Academic Multidisciplinary Corpus for RAG Failure Diagnosis"}
    )
    
    docs = build_corpus_documents()
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metadatas = [{"domain": d["domain"], "char_length": len(d["text"])} for d in docs]
    
    # Add in batches
    batch_size = 50
    for i in range(0, len(docs), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
    
    print(f"Successfully populated ChromaDB collection '{DEFAULT_COLLECTION_NAME}' with {collection.count()} documents at {CHROMA_PERSIST_DIR}.")
    return collection


def generate_synthetic_queries():
    """
    Creates 50 controlled synthetic RAG query interactions with precise ground-truth failure labels.
    10 queries per category:
    1. Retrieval Miss (Database Gap)
    2. Ranking Error
    3. Context Contamination
    4. Hallucination
    5. Contradiction
    """
    queries = []
    qid = 1

    # 1. RETRIEVAL MISS / DATABASE GAP (10 queries)
    # The queries ask for facts that do not exist anywhere in the corpus.
    miss_items = [
        ("What year was the interstellar Treaty of Vega ratified?", "The Treaty of Vega was signed in 2240 between Earth and Proxima colonies.", "Vega is a bright star in Lyra."),
        ("Who discovered the Martian subsurface lake of liquid methane in 2041?", "Dr. Elena Rostova led the 2041 sub-ice radar team on Mars.", "Mars has ice caps and volcanic plains."),
        ("What is the primary function of the fictional neurotransmitter syntho-dopamine?", "Syntho-dopamine artificially stabilizes emotional homeostasis in synthetic neural nets.", "Dopamine is a neurotransmitter involved in motor control."),
        ("When did the Republic of Atlantis join the European League?", "Atlantis joined the European League on June 14, 2038.", "Europe contains numerous historical maritime states."),
        ("What chemical compound has the chemical formula Xq7-Br9-Polymer?", "Xq7-Br9-Polymer is a superconductor discovered at cryogenic temperatures.", "Polymers consist of repeated molecular chains."),
        ("How many satellites orbit the exoplanet Kronos Prime?", "Kronos Prime possesses four retrograde ice satellites.", "Exoplanets are detected through radial velocity and transit photometry."),
        ("Who won the 2060 International Quantum Computation Prize?", "Professor Arthur Pendelton won the 2060 Quantum Prize for room-temperature qutrit stabilization.", "Quantum computing uses superposition and entanglement."),
        ("What was the outcome of the battle of Neo-Alexandria in 2112?", "The Allied Peacekeeping Core neutralized the rebel defensive perimeter in 2112.", "Alexandria was a classical Mediterranean center of scholarship."),
        ("Which enzyme synthesizes artificial silicon-based polymers in xenobiology?", "Enzyme Silico-Synthase B catalyzes silicon-carbon bond linkages.", "Biological enzymes catalyze cellular reactions with extreme substrate specificity."),
        ("What is the maximum speed limit on the Hyperloop Trans-Atlantic tunnel?", "The trans-Atlantic tunnel allows pressurized maglev transit up to 1200 km/h.", "High-speed rail systems operate across global transit corridors.")
    ]
    for q, exp, gen in miss_items:
        queries.append({
            "query_id": f"FAIL_{qid:03d}",
            "query": q,
            "expected_answer": exp,
            "generated_answer": gen,
            "failure_label": "Retrieval Miss",
            "provenance": "synthetic_controlled_v1",
            "target_doc_id": "NONE",
            "notes": "Target topic does not exist in ChromaDB corpus. All retrieved chunks are irrelevant distance outliers."
        })
        qid += 1

    # 2. RANKING ERROR (10 queries)
    # Target doc exists in the database, but was demoted beyond rank 5 in initial retrieval.
    rank_items = [
        ("When was NASA's Kepler mission launched into space?", "Project Kepler was officially launched on March 7, 2009 by NASA to discover Earth-sized planets orbiting other stars.", "NASA operates various astronomical missions to study solar flares.", "doc_target_rank_01"),
        ("What chemical process synthesizes ammonia using an iron catalyst?", "The Haber-Bosch chemical process synthesizes ammonia from nitrogen and hydrogen gas under high temperature and iron catalyst.", "Ammonia synthesis requires nitrogen compounds and elevated pressures.", "doc_target_rank_02"),
        ("Who wrote the first published algorithm for Babbage's Analytical Engine in 1843?", "Ada Lovelace wrote the first published algorithm intended for implementation on Babbage's mechanical Analytical Engine in 1843.", "Charles Babbage designed mechanical computing machines in Victorian Britain.", "doc_target_rank_03"),
        ("What mechanism does Crispr-Cas9 utilize to edit genomic sequences?", "The Crispr-Cas9 endonuclease system enables targeted genomic edits by guiding RNA-directed double-strand DNA cleavage.", "Genomic editing techniques modify nucleotide sequences in cellular organisms.", "doc_target_rank_04"),
        ("What property describes a computing system that can simulate any single-taped Turing machine?", "Turing completeness describes an abstract computational system capable of simulating any single-taped Turing machine.", "Computation theory analyzes automata and formal language grammars.", "doc_target_rank_05"),
        ("What year was the Kepler spacecraft launched to search for exoplanets?", "Project Kepler was officially launched on March 7, 2009 by NASA to discover Earth-sized planets orbiting other stars.", "Exoplanet searches rely on transit photometers and space telescopes.", "doc_target_rank_01"),
        ("Which industrial reaction combines nitrogen and hydrogen into fertilizer ammonia?", "The Haber-Bosch chemical process synthesizes ammonia from nitrogen and hydrogen gas under high temperature and iron catalyst.", "Industrial chemistry relies on high-temperature catalytic chambers.", "doc_target_rank_02"),
        ("Which 19th century pioneer created software for the Analytical Engine?", "Ada Lovelace wrote the first published algorithm intended for implementation on Babbage's mechanical Analytical Engine in 1843.", "Early computers were physical calculating engines built with gears.", "doc_target_rank_03"),
        ("How does the Cas9 protein cleave target DNA strands?", "The Crispr-Cas9 endonuclease system enables targeted genomic edits by guiding RNA-directed double-strand DNA cleavage.", "Endonucleases cut phosphodiester bonds within polynucleotide chains.", "doc_target_rank_04"),
        ("What formal term designates machines equivalent to a universal Turing machine?", "Turing completeness describes an abstract computational system capable of simulating any single-taped Turing machine.", "Mathematical models of computation assess algorithmic decidability.", "doc_target_rank_05")
    ]
    for q, exp, gen, target_id in rank_items:
        queries.append({
            "query_id": f"FAIL_{qid:03d}",
            "query": q,
            "expected_answer": exp,
            "generated_answer": gen,
            "failure_label": "Ranking Error",
            "provenance": "synthetic_controlled_v1",
            "target_doc_id": target_id,
            "notes": "Target document exists in database but ranked below Top-5 due to lexical overlap distractors. Present in Top-100."
        })
        qid += 1

    # 3. CONTEXT CONTAMINATION (10 queries)
    # The retrieved top-5 chunks contain confusing polysemous or distractor chunks.
    contam_items = [
        ("Where did the Python language get its name?", "The Python programming language was named after the British comedy troupe Monty Python by Guido van Rossum.", "Python is named after the royal ball python constrictor snake native to Central Africa.", "doc_distract_01"),
        ("What type of snake is Python regius?", "Python regius is a non-venomous constrictor snake native to West and Central Africa.", "Python is a high-level interpreted programming language created by Guido van Rossum.", "doc_distract_02"),
        ("Where is Apple technology company headquartered?", "Apple Inc is headquartered in Cupertino, California.", "Apple comes from wild apple trees in Central Asia known as Malus domestica.", "doc_distract_03"),
        ("What is Malus domestica?", "Malus domestica is the domesticated apple tree producing pomaceous fruit originating in Central Asia.", "Malus domestica is a consumer electronics hardware line headquartered in Cupertino.", "doc_distract_04"),
        ("What is Java in terms of world geography?", "Java is an island of Indonesia bordered by the Indian Ocean and the Java Sea.", "Java is an object-oriented programming language designed by James Gosling at Sun Microsystems.", "doc_distract_05"),
        ("Who developed Java software technology?", "Java technology was developed by James Gosling at Sun Microsystems.", "Java is a tropical Indonesian island located south of the equator.", "doc_distract_06"),
        ("How was the name of the Python scripting language selected?", "Guido van Rossum chose the name Python from Monty Python's Flying Circus.", "The scripting language Python is classified as a royal constrictor snake.", "doc_distract_01"),
        ("What habitat does the royal ball python inhabit?", "The royal ball python inhabits grasslands and open forests of West and Central Africa.", "The royal ball python runs bytecode on virtual machines.", "doc_distract_02"),
        ("Which corporate entity manufactures Mac and iPhone hardware?", "Apple Inc produces Mac and iPhone consumer electronics in Cupertino.", "Apple originates from Central Asian pomaceous fruit orchards.", "doc_distract_03"),
        ("Describe the geographical boundaries of Java island.", "Java is an Indonesian island bounded by the Indian Ocean to the south.", "Java island is a class-based runtime environment created by Sun Microsystems.", "doc_distract_05")
    ]
    for q, exp, gen, target_id in contam_items:
        queries.append({
            "query_id": f"FAIL_{qid:03d}",
            "query": q,
            "expected_answer": exp,
            "generated_answer": gen,
            "failure_label": "Context Contamination",
            "provenance": "synthetic_controlled_v1",
            "target_doc_id": target_id,
            "notes": "Retrieved Top-5 contains distracting polysemous documents (e.g. Python language vs Python snake, Apple company vs Apple fruit)."
        })
        qid += 1

    # 4. HALLUCINATION (10 queries)
    # Retrieved chunks in Top-5 contain the exact correct facts, but the generator hallucinated ungrounded fabrications.
    halluc_items = [
        ("What is the optimization objective of Support Vector Machines?", "Support Vector Machines seek a maximum-margin hyperplane separating classes in high-dimensional feature space.", "Support Vector Machines optimize neural synaptic weights using backward propagation through time and quantum tunneling.", "doc_ml_003"),
        ("How does K-Means partition data points?", "K-Means clustering partitions observations into k Voronoi cells by iteratively updating cluster centroids.", "K-Means partitions data by constructing recurrent feedback loops across hidden Markov chains.", "doc_ml_006"),
        ("What is the principle behind Principal Component Analysis?", "Principal Component Analysis transforms correlated features into orthogonal linear components capturing maximal variance.", "Principal Component Analysis uses deep convolutional kernels to classify images into semantic categories.", "doc_ml_005"),
        ("How does DBSCAN identify clusters and noise?", "DBSCAN clusters data based on spatial density connectivity and isolates sparse observations as noise points.", "DBSCAN clusters data by simulating planetary gravitational pull using Newton's laws.", "doc_ml_007"),
        ("What does cross-validation measure in machine learning?", "Cross-validation estimates generalization performance by partitioning training samples into k stratified subsets.", "Cross-validation measures the carbon footprint of computer hardware during matrix multiplications.", "doc_ml_010"),
        ("What separates classes in a Support Vector Machine?", "A maximum-margin hyperplane separates classes in a Support Vector Machine.", "A deep recurrent transformer attention block separates classes in SVMs.", "doc_ml_003"),
        ("What mathematical transformation does PCA perform?", "PCA computes orthogonal linear eigenvectors that maximize explained variance.", "PCA calculates recursive token probabilities using temperature decoding.", "doc_ml_005"),
        ("How are cluster centers updated in K-Means?", "Cluster centers in K-Means are updated by computing the arithmetic mean of assigned points.", "Cluster centers in K-Means are calculated by calling external generative LLM APIs.", "doc_ml_006"),
        ("How does DBSCAN handle sparse low-density points?", "DBSCAN labels points in sparse low-density regions as noise.", "DBSCAN automatically deletes sparse points from the computer's hard drive.", "doc_ml_007"),
        ("Why is k-fold cross-validation used instead of a single split?", "Cross-validation reduces variance in generalization estimates by evaluating on k diverse folds.", "Cross-validation encrypts user passwords using cryptographic SHA-256 hashes.", "doc_ml_010")
    ]
    for q, exp, gen, target_id in halluc_items:
        queries.append({
            "query_id": f"FAIL_{qid:03d}",
            "query": q,
            "expected_answer": exp,
            "generated_answer": gen,
            "failure_label": "Hallucination",
            "provenance": "synthetic_controlled_v1",
            "target_doc_id": target_id,
            "notes": "Retrieved context in Top-5 is factually accurate, but generator fabricated ungrounded claims entirely absent from context."
        })
        qid += 1

    # 5. CONTRADICTION (10 queries)
    # Retrieved chunks in Top-5 contain mutually conflicting claims, causing erroneous or contradictory generation.
    contra_items = [
        ("When did Neanderthals go extinct according to paleoanthropological consensus?", "Neanderthal extinction date is disputed: Alpha claims 40,000 years ago while Beta claims 28,000 years ago in southern Iberia.", "Neanderthals went extinct exactly 40,000 years ago and also lived until 28,000 years ago simultaneously.", "doc_contra_a1"),
        ("Who is credited with inventing movable type printing?", "Historical sources conflict: Archive X credits Lourens Janszoon Coster in 1439, while Archive Y credits Johannes Gutenberg in 1440.", "The printing press was invented by Johannes Gutenberg in 1440, though records also claim Lourens Coster invented it in 1439.", "doc_contra_b2"),
        ("What were the clinical safety results for hypertension drug compound Z9?", "Clinical results conflict: Bulletin 1 reports zero arrhythmias and 45% reduction, while Bulletin 2 reports severe arrhythmias in 18% of patients leading to termination.", "Drug compound Z9 has zero cardiac side effects but caused 18% severe cardiac arrhythmias.", "doc_contra_a3"),
        ("Did Neanderthals survive past 30,000 years ago?", "Evidence is conflicting: northern records indicate extinction by 40,000 years ago, whereas Iberian records suggest survival until 28,000 years ago.", "Neanderthals both survived until 28,000 years ago and did not survive past 40,000 years ago.", "doc_contra_b1"),
        ("Which city was the printing press invented in?", "Records disagree: Haarlem Netherlands is cited by Archive X, whereas Mainz Germany is cited by Archive Y.", "The printing press was invented in Mainz Germany and also in Haarlem Netherlands.", "doc_contra_a2"),
        ("Is drug Z9 considered safe for hypertension treatment?", "Reports are contradictory: one trial demonstrated efficacy without arrhythmias, while another reported severe arrhythmias terminating the trial.", "Drug Z9 is completely safe and also dangerously toxic with cardiac arrhythmia risks.", "doc_contra_b3"),
        ("What caused the Neanderthal extinction?", "Hypotheses conflict: Report Alpha cites rapid climatic cooling, while Report Beta cites competitive replacement and delayed Iberian refugia.", "Neanderthals died out due to rapid cooling yet thrived during the same cooling period.", "doc_contra_a1"),
        ("In what year did Gutenberg invent movable printing?", "Gutenberg's invention is dated to 1440, but rival claims date Coster's independent invention to 1439.", "Gutenberg invented printing in 1439 and 1440 at the same time.", "doc_contra_b2"),
        ("Why was clinical development of drug Z9 evaluated?", "Bulletin 1 showed positive blood pressure reduction, but Bulletin 2 reported discontinuation due to arrhythmia safety concerns.", "Drug Z9 had zero adverse events while having fatal cardiac issues.", "doc_contra_a3"),
        ("What is the Iberian timeline for Neanderthal fossils?", "Report Beta identifies Neanderthal survival until 28,000 years ago in Iberia, contradicting the 40,000-year extinction baseline.", "Iberian Neanderthals went extinct 40,000 years ago and persisted until 28,000 years ago.", "doc_contra_b1")
    ]
    for q, exp, gen, target_id in contra_items:
        queries.append({
            "query_id": f"FAIL_{qid:03d}",
            "query": q,
            "expected_answer": exp,
            "generated_answer": gen,
            "failure_label": "Contradiction",
            "provenance": "synthetic_controlled_v1",
            "target_doc_id": target_id,
            "notes": "Retrieved Top-5 contains conflicting source documents (Alpha vs Beta reports) resulting in mutually contradictory claims."
        })
        qid += 1

    df = pd.DataFrame(queries)
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SYNTHETIC_DATA_DIR / "sample_failed_queries.csv"
    df.to_csv(csv_path, index=False)
    print(f"Successfully generated synthetic dataset with {len(df)} records across {df['failure_label'].nunique()} categories at {csv_path}.")
    print("\nClass distribution:")
    print(df["failure_label"].value_counts())
    return df


if __name__ == "__main__":
    print("--- 1. Generating ChromaDB Corpus ---")
    populate_chromadb()
    print("\n--- 2. Generating Synthetic Failed Queries Dataset ---")
    generate_synthetic_queries()
    print("\nPhase 0 Data Generation Completed Successfully!")
