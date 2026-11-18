"""
Demo RAG App #2: Legal Knowledge Base
=======================================
A completely separate RAG application built on a Legal/Law domain corpus.
Uses the SAME ChromaDB default embedding model as our diagnostic platform.

This script:
1. Builds a 120+ document Legal corpus and indexes it into a NEW ChromaDB collection
   stored at data/legal_rag/chroma_db
2. Generates 50 labeled failed queries across all 5 failure categories
3. These simulate real failures of a legal RAG chatbot

After running this script, you plug this new corpus into the diagnostic platform
to show it working on a completely different domain — proving the platform generalizes.
"""

import sys
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — completely separate from the main app's data directory
# ─────────────────────────────────────────────────────────────────────────────
LEGAL_DATA_DIR = PROJECT_ROOT / "data" / "legal_rag"
LEGAL_CHROMA_DIR = LEGAL_DATA_DIR / "chroma_db"
LEGAL_QUERIES_CSV = LEGAL_DATA_DIR / "legal_failed_queries.csv"
LEGAL_COLLECTION_NAME = "legal_knowledge_base"

LEGAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
LEGAL_CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Build the Legal Corpus (120 documents)
# ─────────────────────────────────────────────────────────────────────────────
def build_legal_corpus():
    """
    120 legal knowledge documents across 8 legal sub-domains.
    These simulate chunks from a legal knowledge base a law firm might index.
    """
    docs = []

    # ── Constitutional Law (20 docs) ──────────────────────────────────────────
    constitutional = [
        ("doc_const_001", "Article 14 of the Indian Constitution guarantees equality before law and equal protection of laws to all persons within India."),
        ("doc_const_002", "Article 19 of the Indian Constitution grants six fundamental freedoms including freedom of speech, expression, assembly, association, movement, residence, and profession."),
        ("doc_const_003", "Article 21 of the Indian Constitution provides that no person shall be deprived of their life or personal liberty except according to procedure established by law."),
        ("doc_const_004", "Article 32 of the Indian Constitution grants citizens the right to move the Supreme Court directly for enforcement of Fundamental Rights, also called the right to constitutional remedies."),
        ("doc_const_005", "Article 51A of the Indian Constitution lists eleven fundamental duties that every citizen of India is obligated to perform including respecting the Constitution and national symbols."),
        ("doc_const_006", "The Preamble of the Indian Constitution declares India to be a Sovereign Socialist Secular Democratic Republic and assures justice, liberty, equality, and fraternity."),
        ("doc_const_007", "Article 368 of the Indian Constitution provides the procedure for amendment of the Constitution, requiring a special majority in both Houses of Parliament."),
        ("doc_const_008", "The Directive Principles of State Policy under Part IV of the Indian Constitution are non-justiciable guidelines for the State in policy making to achieve social and economic justice."),
        ("doc_const_009", "Article 44 of the Indian Constitution directs the State to secure a Uniform Civil Code for citizens throughout the territory of India, as a Directive Principle."),
        ("doc_const_010", "The 42nd Constitutional Amendment of 1976 added the words Socialist, Secular, and Integrity to the Preamble and also added Fundamental Duties under Article 51A."),
        ("doc_const_011", "Article 356 of the Indian Constitution allows the President to impose President Rule in a state if the constitutional machinery has failed, also known as President's Rule or State Emergency."),
        ("doc_const_012", "The 73rd Constitutional Amendment of 1992 gave constitutional status to Panchayati Raj institutions and introduced a third tier of government at the village level."),
        ("doc_const_013", "Article 352 of the Indian Constitution empowers the President to proclaim a National Emergency when the security of India is threatened by war, external aggression, or armed rebellion."),
        ("doc_const_014", "The Right to Education Act 2009 was enabled by Article 21A of the Indian Constitution which makes free and compulsory education a Fundamental Right for children aged 6 to 14 years."),
        ("doc_const_015", "Article 143 of the Indian Constitution empowers the President to refer questions of law or fact of public importance to the Supreme Court for advisory opinion."),
        ("doc_const_016", "The doctrine of Basic Structure of the Constitution was established in Kesavananda Bharati v State of Kerala 1973 holding that Parliament cannot amend the basic structure of the Constitution."),
        ("doc_const_017", "Article 123 of the Indian Constitution grants the President the power to promulgate Ordinances when Parliament is not in session, which must be ratified within six weeks of reassembly."),
        ("doc_const_018", "Article 76 of the Indian Constitution provides for the appointment of an Attorney General of India who is the government's chief legal advisor and has the right to audience in all courts."),
        ("doc_const_019", "The Ninth Schedule of the Indian Constitution contains a list of central and state laws that are protected from judicial review regarding violation of fundamental rights."),
        ("doc_const_020", "Article 300A of the Indian Constitution states that no person shall be deprived of his property save by authority of law, added by the 44th Amendment in 1978."),
    ]

    # ── Criminal Law & IPC (20 docs) ─────────────────────────────────────────
    criminal = [
        ("doc_crim_001", "Section 302 of the Indian Penal Code prescribes punishment for murder as death penalty or imprisonment for life and also a fine."),
        ("doc_crim_002", "Section 420 of the Indian Penal Code relates to cheating and dishonestly inducing delivery of property, punishable with imprisonment up to seven years and fine."),
        ("doc_crim_003", "Section 376 of the Indian Penal Code prescribes punishment for rape with rigorous imprisonment of not less than ten years which may extend to life imprisonment."),
        ("doc_crim_004", "Section 304B of the Indian Penal Code defines dowry death as the death of a woman caused by burns, bodily injury, or occurring under suspicious circumstances within seven years of marriage."),
        ("doc_crim_005", "Section 498A of the Indian Penal Code makes cruelty by husband or relatives of husband a cognizable, non-bailable and non-compoundable offence punishable with imprisonment up to three years."),
        ("doc_crim_006", "Section 124A of the Indian Penal Code defines sedition as bringing hatred, contempt, or exciting disaffection towards the government established by law in India."),
        ("doc_crim_007", "The Bharatiya Nyaya Sanhita 2023 replaced the Indian Penal Code 1860 and came into force on July 1, 2024 with new sections renumbered and terrorism-related provisions added."),
        ("doc_crim_008", "Section 307 of the Indian Penal Code relates to attempt to murder which is punishable with imprisonment up to ten years and if hurt is caused the punishment may extend to life imprisonment."),
        ("doc_crim_009", "Section 395 of the Indian Penal Code defines dacoity as robbery committed by five or more persons conjointly, punishable with life imprisonment or rigorous imprisonment up to ten years."),
        ("doc_crim_010", "The Indian Evidence Act 1872 governs the admissibility of evidence in Indian courts and was replaced by the Bharatiya Sakshya Adhiniyam 2023 effective July 1 2024."),
        ("doc_crim_011", "Section 84 of the Indian Penal Code provides that an act done by a person of unsound mind is not an offence if at the time of the act they were incapable of knowing its nature or wrongfulness."),
        ("doc_crim_012", "Section 300 of the Indian Penal Code defines murder as culpable homicide amounting to murder under four specific clauses involving intention and knowledge of causing death."),
        ("doc_crim_013", "The Code of Criminal Procedure 1973 was replaced by the Bharatiya Nagarik Suraksha Sanhita 2023 which came into force on July 1 2024 streamlining criminal procedure."),
        ("doc_crim_014", "Section 320 of the Code of Criminal Procedure lists compoundable offences which can be settled between the parties without court permission or with court permission."),
        ("doc_crim_015", "Section 436A of the Code of Criminal Procedure provides that an undertrial prisoner who has completed half the maximum period of imprisonment for the offence may be released on bail."),
        ("doc_crim_016", "The Prevention of Corruption Act 1988 defines public servants and prescribes penalties for bribery, misappropriation, and possession of disproportionate assets."),
        ("doc_crim_017", "Section 509 of the Indian Penal Code makes it an offence for any person to utter words, make sounds, or gestures intended to insult the modesty of a woman."),
        ("doc_crim_018", "The POCSO Act 2012 provides for protection of children from sexual offences and sexual harassment and uses child-friendly court procedures with special courts."),
        ("doc_crim_019", "Section 34 of the Indian Penal Code provides that when a criminal act is done by several persons in furtherance of common intention all persons are liable as if they did it alone."),
        ("doc_crim_020", "Zero FIR allows a person to file a First Information Report at any police station regardless of jurisdiction, which is then transferred to the appropriate station for investigation."),
    ]

    # ── Contract Law (15 docs) ────────────────────────────────────────────────
    contract = [
        ("doc_cont_001", "Section 10 of the Indian Contract Act 1872 provides that all agreements are contracts if made by free consent of parties competent to contract for a lawful consideration and with lawful object."),
        ("doc_cont_002", "Section 2(h) of the Indian Contract Act 1872 defines a contract as an agreement enforceable by law, distinguishing it from a bare agreement which creates no legal obligation."),
        ("doc_cont_003", "Section 25 of the Indian Contract Act 1872 declares that an agreement without consideration is void subject to exceptions such as natural love, past voluntary service, and time-barred debt."),
        ("doc_cont_004", "Doctrine of Frustration under Section 56 of the Indian Contract Act provides that a contract becomes void when performance becomes impossible due to events beyond the parties control."),
        ("doc_cont_005", "Section 73 of the Indian Contract Act 1872 entitles a party to receive compensation from the defaulting party for loss or damage caused by breach of contract as naturally arising from the breach."),
        ("doc_cont_006", "Specific Relief Act 1963 provides for specific performance of contracts in cases where monetary compensation is inadequate remedy such as contracts relating to immovable property."),
        ("doc_cont_007", "Section 11 of the Indian Contract Act 1872 declares that every person is competent to contract if they are of the age of majority, of sound mind, and not disqualified by any law."),
        ("doc_cont_008", "Consideration under Section 2(d) of the Indian Contract Act is defined as an act abstention or promise by the promisee at the desire of the promisor constituting the price of the promise."),
        ("doc_cont_009", "A voidable contract under Section 2(i) of the Indian Contract Act is one that is enforceable at the option of one party but not at the option of the other, typically arising from coercion or fraud."),
        ("doc_cont_010", "Section 27 of the Indian Contract Act declares void any agreement that restrains a person from exercising a lawful profession, trade, or business, subject to exceptions for sale of goodwill."),
        ("doc_cont_011", "Anticipatory breach of contract occurs when a party declares their intention not to perform their obligation before the due date of performance, entitling the other party to sue immediately."),
        ("doc_cont_012", "An indemnity contract under Section 124 of the Indian Contract Act is one where one party promises to save the other from loss caused by the promisor's conduct or by conduct of a third person."),
        ("doc_cont_013", "Agency under Section 182 of the Indian Contract Act exists when one person the agent is authorised to act for another the principal to create legal relations with third parties."),
        ("doc_cont_014", "Quasi-contracts under Chapter V of the Indian Contract Act impose obligations similar to contracts without an actual agreement, based on principles of unjust enrichment and equity."),
        ("doc_cont_015", "Section 64 of the Indian Contract Act provides that when a voidable contract is rescinded the other party need not perform any promise therein contained."),
    ]

    # ── Property & Land Law (15 docs) ─────────────────────────────────────────
    property_law = [
        ("doc_prop_001", "The Transfer of Property Act 1882 governs transfers of immovable property in India and defines sale, mortgage, lease, exchange, gift, and actionable claim."),
        ("doc_prop_002", "Section 54 of the Transfer of Property Act 1882 defines sale as a transfer of ownership in exchange for a price paid, promised, or part-paid and part-promised."),
        ("doc_prop_003", "Section 58 of the Transfer of Property Act 1882 defines mortgage as a transfer of interest in specific immovable property to secure payment of money advanced or to be advanced."),
        ("doc_prop_004", "The Registration Act 1908 makes registration of documents relating to immovable property mandatory to make them legally valid and to ensure priority of title."),
        ("doc_prop_005", "Section 105 of the Transfer of Property Act 1882 defines lease as a transfer of a right to enjoy immovable property for a certain time in exchange for rent or premium."),
        ("doc_prop_006", "Benami Transactions Prohibition Act 1988 prohibits benami transactions where property is held by one person but actually paid for by another, with intent to defraud."),
        ("doc_prop_007", "The Real Estate Regulation and Development Act 2016 or RERA regulates real estate projects and agents to protect buyers and promote transparency in real estate transactions."),
        ("doc_prop_008", "Section 122 of the Transfer of Property Act 1882 defines gift as a voluntary transfer of property made without consideration by the donor and accepted by the donee."),
        ("doc_prop_009", "Easement under Section 4 of the Indian Easements Act 1882 is a right which the owner of land possesses for the beneficial enjoyment of that land to do some act in relation to other land."),
        ("doc_prop_010", "Section 118 of the Transfer of Property Act 1882 defines exchange as a mutual transfer of ownership of one thing for the ownership of another thing between parties."),
        ("doc_prop_011", "The Specific Relief Act 1963 allows courts to order specific performance of property transfer contracts when monetary damages are inadequate to compensate the aggrieved party."),
        ("doc_prop_012", "Section 60 of the Transfer of Property Act 1882 gives the mortgagor the right to redemption allowing them to reclaim the mortgaged property on repayment of mortgage money."),
        ("doc_prop_013", "The Limitation Act 1963 prescribes time limits for filing suits and applications in courts and bars legal actions filed after the expiration of the limitation period."),
        ("doc_prop_014", "Ancestral property in Hindu Law under Mitakshara school is coparcenary property acquired by a common ancestor and inherited by lineal male descendants up to four generations."),
        ("doc_prop_015", "Section 17 of the Registration Act 1908 makes registration compulsory for documents purporting to create or assign rights in immovable property worth more than one hundred rupees."),
    ]

    # ── Tort Law (15 docs) ────────────────────────────────────────────────────
    tort = [
        ("doc_tort_001", "Negligence as a tort requires proof of duty of care, breach of that duty, causation, and resultant damage to the plaintiff as established in Donoghue v Stevenson 1932."),
        ("doc_tort_002", "Strict liability in tort law holds a person liable for damage caused by their inherently dangerous activity regardless of fault as laid down in Rylands v Fletcher 1868."),
        ("doc_tort_003", "The Motor Vehicles Act 1988 provides for no-fault liability for compensation to victims of motor accidents without proof of negligence under Section 140 and 163A."),
        ("doc_tort_004", "Volenti non fit injuria is a defence in tort law meaning that a person who voluntarily assumes a risk cannot claim damages for harm resulting from that risk."),
        ("doc_tort_005", "Defamation in tort law is the publication of a false statement that lowers a person's reputation in the eyes of right-thinking members of society, divided into libel and slander."),
        ("doc_tort_006", "Contributory negligence reduces the plaintiff's damages in proportion to their own fault in causing the accident under the apportionment approach adopted in Indian courts."),
        ("doc_tort_007", "Vicarious liability makes an employer liable for torts committed by an employee acting within the course of employment even if the employer was not personally at fault."),
        ("doc_tort_008", "Private nuisance in tort law involves unlawful interference with a person's use or enjoyment of land or some right connected with land, requiring substantial and unreasonable interference."),
        ("doc_tort_009", "The Consumer Protection Act 2019 provides for protection of consumer interests and redressal of consumer disputes through District Commissions, State Commissions, and National Commission."),
        ("doc_tort_010", "Absolute liability in Indian tort law established in MC Mehta v Union of India 1987 holds enterprises dealing with hazardous substances absolutely liable without any exceptions."),
        ("doc_tort_011", "Trespass to person includes assault which is creation of reasonable apprehension of force, battery which is actual application of unlawful force, and false imprisonment."),
        ("doc_tort_012", "Remoteness of damage in tort law limits liability to consequences that are not too remote, requiring the type of damage to be reasonably foreseeable as in The Wagon Mound 1961."),
        ("doc_tort_013", "Res ipsa loquitur meaning the thing speaks for itself allows inference of negligence when the accident is of a type that ordinarily does not occur without negligence."),
        ("doc_tort_014", "Injunction as a tort remedy is an equitable remedy ordering a party to stop a continuing wrong such as nuisance, trespass, or defamation, granted at the discretion of the court."),
        ("doc_tort_015", "Malicious prosecution is a tort that occurs when a person is prosecuted without reasonable cause, with malice, and the prosecution terminates in the accused's favour."),
    ]

    # ── Company Law (15 docs) ─────────────────────────────────────────────────
    company = [
        ("doc_comp_001", "The Companies Act 2013 is the primary legislation governing incorporation, management, and dissolution of companies in India, administered by the Ministry of Corporate Affairs."),
        ("doc_comp_002", "A private limited company under the Companies Act 2013 restricts the right to transfer its shares, prohibits invitation to public to subscribe for shares, and limits members to two hundred."),
        ("doc_comp_003", "Section 2(68) of the Companies Act 2013 defines private company as a company having a minimum paid-up share capital as may be prescribed which restricts public participation."),
        ("doc_comp_004", "Corporate veil lifting allows courts to disregard the separate legal personality of a company and hold shareholders or directors personally liable for the company's acts in cases of fraud."),
        ("doc_comp_005", "Section 135 of the Companies Act 2013 mandates Corporate Social Responsibility spending of at least two percent of average net profits for certain class of companies including listed companies."),
        ("doc_comp_006", "Memorandum of Association defines the company's relationship with the outside world including its name, registered office, objects clause, liability clause, and capital clause."),
        ("doc_comp_007", "The Insolvency and Bankruptcy Code 2016 provides a time-bound process for resolving insolvency of companies and individuals through the National Company Law Tribunal."),
        ("doc_comp_008", "Articles of Association contain rules for the internal management of the company including powers of directors, conduct of meetings, and rights of shareholders."),
        ("doc_comp_009", "Section 149 of the Companies Act 2013 requires every listed company to have at least one-third of its total number of directors as independent directors."),
        ("doc_comp_010", "A One Person Company under Section 2(62) of the Companies Act 2013 is a company which has only one member, introduced for the first time in Indian law."),
        ("doc_comp_011", "Derivative action allows a shareholder to bring a suit on behalf of the company against directors or third parties who have harmed the company when the company refuses to sue."),
        ("doc_comp_012", "Section 185 of the Companies Act 2013 restricts companies from giving loans, guarantees, or security to directors or entities in which directors are interested, subject to exceptions."),
        ("doc_comp_013", "SEBI the Securities and Exchange Board of India was established in 1992 to protect investors, develop securities markets, and regulate intermediaries in the Indian securities market."),
        ("doc_comp_014", "Section 230 to 232 of the Companies Act 2013 provide for compromise and arrangement between a company and its creditors or members through the National Company Law Tribunal."),
        ("doc_comp_015", "A Limited Liability Partnership under the LLP Act 2008 combines the benefits of a partnership and a company giving partners limited liability while allowing flexible management structure."),
    ]

    # ── International Law (10 docs) ───────────────────────────────────────────
    intl = [
        ("doc_intl_001", "The United Nations Charter signed in 1945 established the UN and its principal organs including the General Assembly, Security Council, and International Court of Justice."),
        ("doc_intl_002", "The Vienna Convention on the Law of Treaties 1969 codifies the law governing treaties between states including their formation, validity, interpretation, and termination."),
        ("doc_intl_003", "Customary international law consists of practices that states follow out of a sense of legal obligation and are binding on all states including those that did not consent."),
        ("doc_intl_004", "The principle of Sovereign Immunity holds that a foreign state cannot be sued in the domestic courts of another state for acts performed in its governmental capacity."),
        ("doc_intl_005", "The Rome Statute 1998 established the International Criminal Court with jurisdiction over genocide, war crimes, crimes against humanity, and the crime of aggression."),
        ("doc_intl_006", "UNCLOS the United Nations Convention on the Law of the Sea 1982 defines the rights and responsibilities of nations with respect to their use of the world's oceans."),
        ("doc_intl_007", "The principle of non-refoulement under the 1951 Refugee Convention prohibits states from returning refugees to territories where they face serious threats to their life or freedom."),
        ("doc_intl_008", "Jus cogens norms or peremptory norms of international law are fundamental principles from which no derogation is permitted, including prohibitions on genocide, slavery, and torture."),
        ("doc_intl_009", "Bilateral Investment Treaties create legally binding obligations between two states to promote and protect investments made by nationals of each state in the other state's territory."),
        ("doc_intl_010", "The Arbitration and Conciliation Act 1996 incorporates UNCITRAL Model Law and provides for domestic and international commercial arbitration as an alternative dispute resolution mechanism."),
    ]

    # ── Contradiction Corpus (conflicting docs for Contradiction failure) ─────
    contradiction_docs = [
        ("doc_contra_01a", "Section 377 of the Indian Penal Code criminalizes consensual same-sex relations between adults and was upheld as constitutional by the Supreme Court in Suresh Kumar Koushal v Naz Foundation 2013."),
        ("doc_contra_01b", "Section 377 of the Indian Penal Code was read down by the Supreme Court in Navtej Singh Johar v Union of India 2018 decriminalizing consensual same-sex relations between adults."),
        ("doc_contra_02a", "The Supreme Court in Indra Sawhney v Union of India 1992 upheld the 27 percent reservation for OBCs but imposed a 50 percent ceiling on total reservations in government jobs."),
        ("doc_contra_02b", "The 103rd Constitutional Amendment 2019 introduced 10 percent reservation for Economically Weaker Sections in government jobs and educational institutions, which the Supreme Court upheld in 2022, effectively making total reservation 60 percent in several states."),
        ("doc_contra_03a", "Legal scholars argue that the right to privacy was not a fundamental right under the Indian Constitution as held in MP Sharma 1954 and Kharak Singh 1962 Supreme Court decisions."),
        ("doc_contra_03b", "The Supreme Court in Justice KS Puttaswamy v Union of India 2017 held unanimously that the right to privacy is a fundamental right under Article 21 of the Indian Constitution, overruling MP Sharma and Kharak Singh."),
    ]

    for domain, domain_docs in [
        ("Constitutional Law", constitutional),
        ("Criminal Law", criminal),
        ("Contract Law", contract),
        ("Property Law", property_law),
        ("Tort Law", tort),
        ("Company Law", company),
        ("International Law", intl),
        ("Contradiction Source", contradiction_docs),
    ]:
        for doc_id, text in domain_docs:
            docs.append({"id": doc_id, "text": text, "domain": domain})

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Index into ChromaDB
# ─────────────────────────────────────────────────────────────────────────────
def populate_legal_chromadb():
    client = chromadb.PersistentClient(path=str(LEGAL_CHROMA_DIR))

    try:
        client.delete_collection(name=LEGAL_COLLECTION_NAME)
        print(f"Deleted existing collection '{LEGAL_COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=LEGAL_COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "description": "Legal Knowledge Base for RAG Failure Diagnostic Demo"
        }
    )

    docs = build_legal_corpus()
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metadatas = [{"domain": d["domain"], "char_length": len(d["text"])} for d in docs]

    batch_size = 50
    for i in range(0, len(docs), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    print(f"✅ Legal ChromaDB populated: {collection.count()} documents at {LEGAL_CHROMA_DIR}")
    return collection


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Generate 50 Labeled Failed Queries
# ─────────────────────────────────────────────────────────────────────────────
def generate_legal_failed_queries():
    queries = []
    qid = 1

    # ── 10 × RETRIEVAL MISS ──────────────────────────────────────────────────
    # Topic does not exist in legal corpus
    retrieval_miss = [
        ("What are the tax rates under the fictional GST Amendment Act of 2029?", "The 2029 GST Amendment introduced tiered rates of 8, 14, and 24 percent.", "GST is a consumption-based tax applicable in India."),
        ("What is the penalty under the Cyber-Criminal Prosecution Act 3000?", "The Cyber-Criminal Prosecution Act 3000 prescribes capital punishment for data theft.", "Cybercrime is handled under the IT Act."),
        ("Explain the provisions of the Mars Jurisdiction Treaty of 2041.", "The Mars Jurisdiction Treaty extends Indian law to Indian nationals on Mars.", "Space law is governed by the Outer Space Treaty."),
        ("What does the fictional Quantum Property Rights Act of 2050 state?", "The Quantum Property Rights Act grants ownership of quantum states to their observers.", "Property rights are governed by the Transfer of Property Act 1882."),
        ("What is the punishment under Section 999 of the Indian Penal Code?", "Section 999 prescribes imprisonment for misuse of artificial intelligence systems.", "The IPC contains provisions up to Section 511."),
        ("What are the provisions of the Interstellar Commerce Act 2060?", "The Interstellar Commerce Act 2060 regulates trade between planetary systems.", "International trade is governed by WTO agreements."),
        ("Explain Article 500 of the Indian Constitution.", "Article 500 grants citizens the right to algorithmic fairness in judicial decisions.", "The Indian Constitution has 395 articles in its original form."),
        ("What is prescribed under the Biodiversity Patent Act of 2045?", "The Biodiversity Patent Act of 2045 allows patenting of entire ecosystems.", "Patents are governed by the Patents Act 1970."),
        ("What does the Neural Rights Act 2033 protect?", "The Neural Rights Act 2033 protects citizens from non-consensual brain scanning.", "Mental health is covered under the Mental Healthcare Act 2017."),
        ("Explain the provisions of the Anti-Deepfake Evidence Act 2035.", "The Anti-Deepfake Evidence Act 2035 makes deepfakes inadmissible as evidence.", "Evidence is governed by the Indian Evidence Act 1872."),
    ]
    for q, exp, gen in retrieval_miss:
        queries.append({"query_id": f"LFAIL_{qid:03d}", "query": q, "expected_answer": exp, "generated_answer": gen,
                        "failure_label": "Retrieval Miss", "provenance": "legal_rag_v1", "target_doc_id": "NONE",
                        "notes": "Query topic absent from legal corpus."})
        qid += 1

    # ── 10 × RANKING ERROR ───────────────────────────────────────────────────
    # Correct doc exists but demoted past rank 5
    ranking_error = [
        ("What does Article 14 of the Indian Constitution guarantee?", "Article 14 guarantees equality before law and equal protection of laws to all persons within India.", "The Constitution provides various fundamental rights to citizens including right to life and liberty.", "doc_const_001"),
        ("What is the punishment for rape under Section 376 IPC?", "Section 376 IPC prescribes rigorous imprisonment of not less than ten years which may extend to life imprisonment.", "Sexual offences are dealt with stringently under Indian law with heavy penalties.", "doc_crim_003"),
        ("What does Section 10 of the Indian Contract Act provide?", "Section 10 provides that all agreements are contracts if made by free consent of parties competent to contract for a lawful consideration and with a lawful object.", "The Indian Contract Act 1872 is a central legislation governing contracts and agreements.", "doc_cont_001"),
        ("What is the definition of sale under the Transfer of Property Act?", "Section 54 defines sale as a transfer of ownership in exchange for a price paid, promised, or part-paid and part-promised.", "Property transactions in India are governed by several laws including the Transfer of Property Act.", "doc_prop_002"),
        ("What is negligence in tort law?", "Negligence as a tort requires proof of duty of care, breach of that duty, causation, and resultant damage as established in Donoghue v Stevenson 1932.", "Tort law governs civil wrongs and provides remedies for damage caused by the wrongful act of another.", "doc_tort_001"),
        ("What is the Companies Act 2013?", "The Companies Act 2013 is the primary legislation governing incorporation, management, and dissolution of companies in India, administered by the Ministry of Corporate Affairs.", "Company law in India has undergone significant reforms over the past two decades.", "doc_comp_001"),
        ("What is the Bharatiya Nyaya Sanhita?", "The Bharatiya Nyaya Sanhita 2023 replaced the Indian Penal Code 1860 and came into force on July 1 2024 with new sections renumbered and terrorism-related provisions added.", "India reformed its criminal laws with three new codes in 2023 replacing colonial era legislation.", "doc_crim_007"),
        ("What is the doctrine of frustration in contract law?", "Doctrine of Frustration under Section 56 of the Indian Contract Act provides that a contract becomes void when performance becomes impossible due to events beyond the parties control.", "When circumstances change fundamentally a contract may be discharged under certain conditions.", "doc_cont_004"),
        ("What is strict liability in tort law?", "Strict liability in tort law holds a person liable for damage caused by their inherently dangerous activity regardless of fault as laid down in Rylands v Fletcher 1868.", "Liability in tort can arise without proof of fault in certain categories of dangerous activities.", "doc_tort_002"),
        ("What is a private limited company?", "A private limited company under the Companies Act 2013 restricts the right to transfer its shares, prohibits invitation to public to subscribe for shares, and limits members to two hundred.", "Companies in India are classified into public and private companies with different regulations.", "doc_comp_002"),
    ]
    for q, exp, gen, target in ranking_error:
        queries.append({"query_id": f"LFAIL_{qid:03d}", "query": q, "expected_answer": exp, "generated_answer": gen,
                        "failure_label": "Ranking Error", "provenance": "legal_rag_v1", "target_doc_id": target,
                        "notes": "Target doc exists in DB but was demoted beyond rank 5."})
        qid += 1

    # ── 10 × CONTEXT CONTAMINATION ───────────────────────────────────────────
    # Wrong domain retrieved — e.g. query about constitutional law retrieved criminal law docs
    contamination = [
        ("What does Article 21 protect?", "Article 21 provides that no person shall be deprived of their life or personal liberty except according to procedure established by law.", "Section 21 of the Indian Contract Act declares that a mistake as to law does not render a contract voidable."),
        ("What is an easement in property law?", "Easement under Section 4 of the Indian Easements Act 1882 is a right which the owner of land possesses for beneficial enjoyment allowing some act over other land.", "Easement in criminal law refers to the procedural requirement to record the statement of a witness."),
        ("Explain the concept of agency in contract law.", "Agency under Section 182 exists when one person the agent is authorised to act for another the principal to create legal relations with third parties.", "Agency in criminal law refers to the arrangement where police act as agents of the state in investigation."),
        ("What is defamation in law?", "Defamation in tort law is the publication of a false statement that lowers a person's reputation in the eyes of right-thinking members of society, divided into libel and slander.", "Defamation under Section 499 IPC is an offence committed by imputing something to a person to harm their reputation in public."),
        ("What is the right to redemption in mortgage?", "Section 60 of the Transfer of Property Act gives the mortgagor the right to redemption allowing them to reclaim mortgaged property on repayment of mortgage money.", "Redemption in company law refers to the buyback of preference shares by a company using profits or fresh issue proceeds."),
        ("What is an injunction as a legal remedy?", "Injunction as a tort remedy is an equitable remedy ordering a party to stop a continuing wrong such as nuisance, trespass, or defamation, granted at the discretion of the court.", "Injunction in constitutional law is a writ issued by a High Court restraining a public authority from acting unconstitutionally."),
        ("Explain the doctrine of basic structure.", "The doctrine of Basic Structure was established in Kesavananda Bharati v State of Kerala 1973 holding that Parliament cannot amend the basic structure of the Constitution.", "Basic structure in contract law refers to the essential terms of a contract without which the contract cannot be performed."),
        ("What is res ipsa loquitur?", "Res ipsa loquitur meaning the thing speaks for itself allows inference of negligence when the accident is of a type that ordinarily does not occur without negligence.", "Res ipsa loquitur in criminal law shifts the burden of proof to the accused when certain facts are proven by the prosecution."),
        ("What is the Preamble to the Indian Constitution?", "The Preamble declares India to be a Sovereign Socialist Secular Democratic Republic and assures justice, liberty, equality, and fraternity.", "The preamble of a contract is the introductory statement that sets out the parties, recitals, and background to the agreement."),
        ("What is SEBI?", "SEBI the Securities and Exchange Board of India was established in 1992 to protect investors, develop securities markets, and regulate intermediaries in the Indian securities market.", "SEBI stands for Supervision of Employer Benefit Insurance and handles insurance claims regulation under a 1994 statute."),
    ]
    for q, exp, gen in contamination:
        queries.append({"query_id": f"LFAIL_{qid:03d}", "query": q, "expected_answer": exp, "generated_answer": gen,
                        "failure_label": "Context Contamination", "provenance": "legal_rag_v1", "target_doc_id": "NONE",
                        "notes": "Wrong legal sub-domain retrieved — cross-domain contamination."})
        qid += 1

    # ── 10 × HALLUCINATION ───────────────────────────────────────────────────
    # LLM generates facts not in corpus at all
    hallucination = [
        ("What is the punishment for murder under Section 302 IPC?", "Section 302 IPC prescribes punishment for murder as death penalty or imprisonment for life and also a fine.", "Section 302 IPC prescribes punishment for murder as a mandatory death sentence combined with forfeiture of all property of the accused."),
        ("What is Article 32 of the Indian Constitution?", "Article 32 grants citizens the right to move the Supreme Court directly for enforcement of Fundamental Rights, also called the right to constitutional remedies.", "Article 32 grants citizens the right to petition the President of India directly for enforcement of rights, with a 30-day response mandate."),
        ("What is the Specific Relief Act?", "Specific Relief Act 1963 provides for specific performance of contracts in cases where monetary compensation is inadequate remedy such as contracts relating to immovable property.", "Specific Relief Act 1963 allows courts to enforce contracts by compelling the defendant to pay triple damages as a form of punitive specific relief."),
        ("What is dowry death under Section 304B IPC?", "Section 304B defines dowry death as the death of a woman caused by burns, bodily injury, or occurring under suspicious circumstances within seven years of marriage.", "Section 304B defines dowry death as death occurring within three years of marriage and requires DNA evidence linking dowry demand to the death."),
        ("What is the minimum membership requirement for a private company?", "A private limited company under the Companies Act 2013 restricts membership to two hundred and limits share transfers.", "A private limited company requires a minimum of five members and must have at least two independent directors on its board at all times."),
        ("What does the Rome Statute establish?", "The Rome Statute 1998 established the International Criminal Court with jurisdiction over genocide, war crimes, crimes against humanity, and the crime of aggression.", "The Rome Statute 1998 established the International Human Rights Tribunal and created the office of UN Human Rights Prosecutor with veto power over national prosecutions."),
        ("What is absolute liability in Indian tort law?", "Absolute liability established in MC Mehta v Union of India 1987 holds enterprises dealing with hazardous substances absolutely liable without any exceptions.", "Absolute liability in Indian law was introduced in the Environment Protection Amendment Act 1991 and applies only to public sector undertakings."),
        ("What is UNCLOS?", "UNCLOS the United Nations Convention on the Law of the Sea 1982 defines the rights and responsibilities of nations with respect to their use of the world's oceans.", "UNCLOS is the United Nations Convention on Land Occupation Sovereignty signed in 1992 that grants coastal nations permanent sovereignty over 500 nautical miles."),
        ("What does Section 25 of the Indian Contract Act state?", "Section 25 declares that an agreement without consideration is void subject to exceptions such as natural love, past voluntary service, and time-barred debt.", "Section 25 declares that all agreements require monetary consideration and that non-monetary consideration such as love or affection is legally invalid in all circumstances."),
        ("What is the right to privacy in India?", "The Supreme Court in Justice KS Puttaswamy v Union of India 2017 held unanimously that the right to privacy is a fundamental right under Article 21 of the Indian Constitution.", "The right to privacy was introduced as a fundamental right by the 86th Constitutional Amendment in 2019 and is explicitly listed under Article 21B of the Constitution."),
    ]
    for q, exp, gen in hallucination:
        queries.append({"query_id": f"LFAIL_{qid:03d}", "query": q, "expected_answer": exp, "generated_answer": gen,
                        "failure_label": "Hallucination", "provenance": "legal_rag_v1", "target_doc_id": "NONE",
                        "notes": "Generated answer contains fabricated facts not in corpus."})
        qid += 1

    # ── 10 × CONTRADICTION ───────────────────────────────────────────────────
    # Corpus has conflicting docs, LLM synthesizes contradictory answer
    contradiction = [
        ("Is Section 377 IPC constitutional or unconstitutional?", "Section 377 was read down in Navtej Singh Johar 2018 decriminalizing consensual same-sex acts, overruling the 2013 Koushal judgment that upheld it.", "Section 377 is both constitutional as held in Koushal 2013 and unconstitutional as held in Navtej Johar 2018 simultaneously creating legal uncertainty."),
        ("What is the reservation ceiling in India after 2022?", "The 50 percent ceiling from Indra Sawhney 1992 and the 10 percent EWS reservation upheld in 2022 create a contested total reservation landscape.", "Reservations cannot exceed 50 percent per Indra Sawhney, but the Supreme Court in 2022 also upheld 10 percent EWS taking totals beyond 50 percent in many states."),
        ("Is privacy a fundamental right under the Indian Constitution?", "Privacy is a fundamental right under Article 21 as held in Puttaswamy 2017, overruling MP Sharma 1954 and Kharak Singh 1962 which held otherwise.", "Legal scholars cite MP Sharma and Kharak Singh to argue privacy is not a fundamental right, while Puttaswamy 2017 declares it is, making the position contradictory across time."),
        ("When was Section 377 decriminalized?", "Consensual same-sex relations were decriminalized in 2018 by the Navtej Johar judgment of the Supreme Court of India.", "Section 377 was decriminalized in 2009 by the Delhi High Court but re-criminalized in 2013 and finally decriminalized again in 2018 by the Supreme Court."),
        ("Does India have a uniform civil code?", "Article 44 of the Indian Constitution directs the State to secure a Uniform Civil Code but it has not been implemented uniformly across India.", "Goa has a Uniform Civil Code applicable to all residents, while other states follow personal laws, creating a contradictory situation on UCC implementation."),
        ("What is the total reservation percentage in government jobs in India?", "The Indra Sawhney judgment capped reservations at 50 percent, applicable to OBC reservations of 27 percent plus SC/ST reservations.", "Total reservations in India are 49.5 percent under the SC/ST/OBC categories plus 10 percent EWS taking the constitutional total to 59.5 percent in central government jobs."),
        ("What is the limitation period for filing a civil suit in India?", "The Limitation Act 1963 prescribes varying limitation periods from 1 to 30 years depending on the type of suit and the nature of the right.", "Civil suits must generally be filed within three years under the Limitation Act, though suits on immovable property allow up to twelve years creating varied periods."),
        ("Is consent a valid defence under Section 376 IPC?", "Section 376 IPC prescribes punishment for rape and consent is a key element, with marital rape not recognized as rape under the original IPC.", "Consent is both a defence and not a defence under rape law because while consent prevents conviction for rape, marital rape is contested with conflicting court orders."),
        ("What is the constitutional status of EWS reservation?", "The 103rd Constitutional Amendment introducing 10 percent EWS reservation was upheld by the Supreme Court by a 3:2 majority in November 2022.", "Three judges upheld EWS reservation while two judges held it unconstitutional for violating the basic structure, creating a split decision with contested implications."),
        ("Is Goa the only state with a Uniform Civil Code?", "Goa follows the Goa Civil Code also called the Portuguese Civil Code which is a form of uniform civil code applicable to all residents regardless of religion.", "While Goa has a civil code, Uttarakhand enacted its own Uniform Civil Code in 2024 making it the first state in independent India to implement a UCC after independence."),
    ]
    for q, exp, gen in contradiction:
        queries.append({"query_id": f"LFAIL_{qid:03d}", "query": q, "expected_answer": exp, "generated_answer": gen,
                        "failure_label": "Contradiction", "provenance": "legal_rag_v1", "target_doc_id": "NONE",
                        "notes": "Conflicting legal sources retrieved — model synthesizes contradictory answer."})
        qid += 1

    return queries


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Save CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_legal_queries(queries):
    fieldnames = ["query_id", "query", "expected_answer", "generated_answer",
                  "failure_label", "provenance", "target_doc_id", "notes"]
    with open(LEGAL_QUERIES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(queries)
    print(f"✅ Legal failed queries CSV saved: {LEGAL_QUERIES_CSV} ({len(queries)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("LEGAL RAG APP — Corpus Builder & Failed Query Generator")
    print("=" * 70)

    print("\n--- Step 1: Building Legal Corpus & Indexing into ChromaDB ---")
    populate_legal_chromadb()

    print("\n--- Step 2: Generating 50 Labeled Legal RAG Failed Queries ---")
    queries = generate_legal_failed_queries()
    save_legal_queries(queries)

    # Verify class distribution
    from collections import Counter
    dist = Counter(q["failure_label"] for q in queries)
    print("\nClass distribution:")
    for label, count in dist.items():
        print(f"  {label}: {count}")

    print("\n--- Step 3: Verifying ChromaDB connection ---")
    client = chromadb.PersistentClient(path=str(LEGAL_CHROMA_DIR))
    col = client.get_collection(LEGAL_COLLECTION_NAME)
    print(f"  Collection: '{LEGAL_COLLECTION_NAME}' — {col.count()} documents")

    print("\n" + "=" * 70)
    print("✅ Legal RAG App Ready!")
    print(f"   ChromaDB Path : {LEGAL_CHROMA_DIR}")
    print(f"   Collection    : {LEGAL_COLLECTION_NAME}")
    print(f"   Query CSV     : {LEGAL_QUERIES_CSV}")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Run feature extraction: python scripts/run_legal_pipeline.py")
    print("  2. Models will retrain on legal data automatically")
    print("  3. Open http://localhost:8501 — Legal RAG demo is live!")
