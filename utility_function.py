from bs4 import BeautifulSoup
import requests
import re
import json
import os
import html
import time
from urllib.parse import urljoin
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

#Check crossref email
CROSSREF_EMAIL = os.getenv('CROSSREF_EMAIL') #Crossref recommends identifying your application with an email through the mailto parameter or user-agent

#This identifies the application with external servers
HEADERS = {
    "User-Agent": (
        "AutomatedLiteratureMining/0.1 "
        f"(mailto:{CROSSREF_EMAIL})"
    ),
    "Accept": ( #Tells the remote server which response formats it can process
        "text/html,application/xhtml+xml,"
        "application/json;q=0.9,*/*;q=0.8"
    ),
}

##########################################################################################################################################
#Load environment variables
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
ollama_api_key = os.getenv('OLLAMA_API_KEY')

#Load model URLs from environment variables in a file called .env
ANTHROPIC_BASE_URL = os.getenv('ANTHROPIC_BASE_URL')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')

#Assign variables to the models
QUERY_GENERATION_MODEL = "anthropic/claude-sonnet-5"
PUBLICATION_EXTRACTION_MODEL = {"model": "openai/gpt-4.1-mini",
                                "reasoning": {
                                    "effort": "medium"
                                },
                                "temperature": 0
                                }

# Connect to OpenAI, Anthropic and Ollama
openrouter = OpenAI(
    base_url=ANTHROPIC_BASE_URL,
    api_key=anthropic_api_key,
    timeout=120.0,
)

ollama = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=ollama_api_key,
    timeout=120.0,
)

##########################################################################################################################################
############################################### General LLM call function ################################################################
def call_chat_model( #Prevents every LLM yask from rewritting API logic
    client,
    model,
    system_prompt,
    user_prompt,
    temperature=0,
    reasoning=None):
    """
    Constructs the standard request
    Send a request to an OpenAI-compatible chat-completions API.
    """
    request = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt, #Determines the model's role and behavioral constraints
            },
            {
                "role": "user",
                "content": user_prompt, #Provides the specific instance work
            },
        ],
        "temperature": temperature,
    }

    extra_body = {} #Allows the publication content extraction model to use reasoning while the query model does not.

    if reasoning is not None:
        extra_body["reasoning"] = reasoning
    
    response = client.chat.completions.create(
        **request,
        extra_body=extra_body,
    )

    content = response.choices[0].message.content

    if not content: #Model communication layer---don't let an empty model response silently continue downstream
        raise RuntimeError(
            f"Model '{model}' returned an empty response."
        )
    return content.strip()

#########################################################################################################################################
####################### Function to Parse LLM JSON Output and Convert it into Python object (dictionary)#################################
#Small open models sometimes wrap JSON in Markdown fences or add explanatory text
def parse_json_response(content): #The function transforms the argument content which is a JSON string into a Python dictionary
    """
    Parse JSON returned by an LLM.
    Handles:
    - plain JSON; #Standard JSON text
    - ```json code fences; #JSON wrapped in Markdown fences
    - small amounts of text surrounding a JSON object. #Text around JSON
    """
    if not content: #Checks whether content is empty or otherwise evaluates to False
        raise ValueError("The model returned an empty response.")

    cleaned = content.strip() #Removes whitespace from the beginning and end of the JSON string
    cleaned = re.sub(
        r"^```\s*(?:json)?\s*", #Remove a Markdown code fence from the beginning of the JSON string and replace with nothing.
        "",
        cleaned,
        flags=re.IGNORECASE) #Makes the match case-insensitive
    cleaned = re.sub(r"\s*```$", "", cleaned) #Removes a closing Markdown fence from the end of the response

    try:
        return json.loads(cleaned) #Converts a JSON-formatted string into a Python object
    except json.JSONDecodeError: #Raise the error encountered and continue to next line without doing anything
        pass

    object_match = re.search( #Use this fallback extraction strategy when the response is not valid JSON
        r"\{.*\}", #Searches for text beginning with { and ending with }
        cleaned,
        flags=re.DOTALL) #To ensure the search does not stop at the first newline and fail to reach the closing brace

    if object_match:
        return json.loads(object_match.group(0)) #If brace-delimited section is found, extracts the matched text and parses it with json.loads()
    raise ValueError(
        "The model response did not contain valid JSON:\n"
        f"{content}"
    )

##########################################################################################################################################
########################## Function to convert messy, inconsistent web text into a standardized representation ###########################
def clean_whitespace(text):
    """
    Normalize excessive spaces and blank lines.

    Parameters
    ----------
    text : str | None
        Text to clean.

    Returns
    -------
    str | None
        Cleaned text, or None when no text was supplied.
    """
    if not text: #If text is empty return None
        return None

    text = html.unescape(str(text)) #Convert text to string and handle HTML entities such as convert "Tom &amp; Jerry" into "Tom & Jerry"
    text = re.sub(r"[ \t]+", " ", text) #Cleanse up repeat spaces or tabs into single space
    text = re.sub(r"\n[ \t]+", "\n", text) #Cleanse up leading spaces/indentation after a newlines
    text = re.sub(r"\n{3,}", "\n\n", text) #Retain a single blank line where there are three or more lines (reduces excessive blank lines)

    return text.strip() #Cleanse up whitespace only at the beginning and end of the string

########################################################################################################################################################################
################################################## Query Generation With Model 1 #######################################################################################
#Tansform scientific topic into a small, structured set of focused search queries that Python retrieval layer can later submit to PubMed, or another literature database
def generate_search_queries(topic, maximum_queries=6): #Function takes two arguments: Scientific topic and the max number of search queries to return
    """
    Use Model 1 to expand a scientific topic into focused queries.

    Model 1 does not perform the search itself. It creates the
    search strategy used by the Python search function.
    """
    system_prompt = """
You are a scientific literature search strategist with expertise in molecular biology, 
cancer biology, cell and gene therapy, genomics, bioinformatics, and biomedical research.

Generate focused search queries for discovering peer-reviewed research
articles related to the user's topic.

Requirements:
1. Preserve the user's central scientific question.
2. Include useful synonyms and closely related terminology.
3. Do not make the queries excessively broad.
4. Prefer queries that describe the biological topic, analytical method,
   disease, sample type, or sequencing technology when relevant.
5. Do not generate more queries than requested.
6. Return valid JSON only.
7. Do not include Markdown or explanations outside the JSON.

Return this structure:
{
  "original_topic": "the original topic",
  "search_queries": [
    "query one",
    "query two"
  ],
  "required_concepts": [ #Generates concepts (Key words in the topic) that define relevance
    "concept one"
  ]
}
"""

    user_prompt = f"""
Research topic:
{topic}

Maximum number of queries:
{maximum_queries}
"""
    raw_response = call_chat_model( #Sends the prompts to the model through the wrapper function
        client=openrouter,
        model=QUERY_GENERATION_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0, #This asks for low-randomness output
    )

    result = parse_json_response(raw_response) #Converts the response from JSON text into a Python dictionary

    queries = result.get("search_queries", []) #Return the value associated with "search_queries" if it exists in the JSON

    queries = [ #This list comprehension cleans each query and removes blank or unusable entries
        clean_whitespace(query)
        for query in queries
        if clean_whitespace(query)
    ]
    #Removes case-insensitive duplicates while retaining the original formatting of the first occurrence
    seen = set()
    unique_queries = []

    for query in queries:
        key = query.casefold()

        if key not in seen:
            seen.add(key)
            unique_queries.append(query)
    #unique_queries = list(dict.fromkeys(queries)) #Remove duplicates while preserving order.

    if not unique_queries: #Checks whether the model produced at least one valid query after cleaning and deduplication
        raise ValueError(
            "Model 1 did not generate any usable queries."
        )

    return {
        "original_topic": topic,
        "search_queries": unique_queries[:maximum_queries],
        "required_concepts": result.get(
            "required_concepts",
            [],
        ),
    }

##########################################################################################################################################
############################################# Publication Relevance Filtering With Model 1 ###############################################
def classify_article_relevance(topic, publication):
    """
    Ask Model 1 whether a candidate publication directly matches
    the research topic.
    """
    system_prompt = """
You are a scientific literature relevance classifier.

Determine whether the candidate publication is directly relevant to the
research topic.

Rules:
1. Use only the supplied title, abstract, journal, and metadata.
2. Do not use outside knowledge.
3. Do not mark an article relevant merely because it contains one
   matching word.
4. Consider whether the objective, methods, dataset, results, or
   conclusions directly address the topic.
5. When the abstract is missing, be conservative.
6. The relevance score must be between 0 and 1.
7. Return valid JSON only.
8. Do not include Markdown.

Return:

{
  "is_relevant": true,
  "relevance_score": 0.85,
  "reason": "Brief evidence-based explanation",
  "matched_concepts": [
    "matched scientific concept"
  ]
}
"""

    user_prompt = f"""
Research topic:
{topic}

Candidate publication:

Title:
{publication.get("title") or "Not available"}

Journal:
{publication.get("journal") or "Not available"}

Publication date:
{publication.get("publication_date") or "Not available"}

Abstract:
{publication.get("abstract") or "Not available"}
"""

    raw_response = call_chat_model(
        client=openrouter,
        model=QUERY_GENERATION_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0,
    )

    result = parse_json_response(raw_response)

    score = result.get("relevance_score", 0)

    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0

    score = max(0.0, min(score, 1.0))

    return {
        "is_relevant": bool(
            result.get("is_relevant", False)
        ),
        "relevance_score": score,
        "reason": clean_whitespace(
            result.get("reason")
        ),
        "matched_concepts": result.get(
            "matched_concepts",
            [],
        ),
    }

##########################################################################################################################################
################################################ Function to strip HTML from Crossref abstracts ##########################################
def strip_html_tags(text): #Uses BeautifulSoup to convert an abstract containing markup into plain text making it suitable for the LLM classification and downstream storage
    """
    Remove HTML or XML tags from a text value.
    """
    if not text:
        return None

    soup = BeautifulSoup(text, "html.parser")

    return clean_whitespace(
        soup.get_text(separator=" ", strip=True)
    )

############################################################################################################################################
#################################### Function to Standardize Different DOIs into One Consistent Format #####################################
def normalize_doi(doi):
    """
    Convert DOI representations to a normalized lowercase DOI.

    Examples
    --------
    https://doi.org/10.1016/j.cell.2025.01.001
    doi:10.1016/j.cell.2025.01.001

    Both become:

    10.1016/j.cell.2025.01.001
    """
    if not doi:
        return None

    doi = doi.strip().lower() #Remove surrounding whitespace and convert to lowercase

    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ]

    for prefix in prefixes:
        if doi.startswith(prefix):
            doi = doi[len(prefix):] #Removes match prefix using slicing and returns everything beginning immediately after the prefix

    return doi.strip()

#############################################################################################################################################
############################################# Converts DOI into URL #########################################################################
def doi_to_url(doi):
    """
    Convert a DOI into a resolving URL.
    """
    normalized_doi = normalize_doi(doi)

    if not normalized_doi:
        return None

    return f"https://doi.org/{normalized_doi}"

#############################################################################################################################################
###################################################### Convert a Crossref date object #######################################################
def crossref_date_to_string(date_object):
    """
    Convert a Crossref date-parts object to an ISO-like date string.
    """
    if not date_object:
        return None

    date_parts = date_object.get("date-parts")

    if not date_parts or not date_parts[0]:
        return None

    parts = date_parts[0]

    if len(parts) == 3:
        year, month, day = parts
        return f"{year:04d}-{month:02d}-{day:02d}"

    if len(parts) == 2:
        year, month = parts
        return f"{year:04d}-{month:02d}"

    if len(parts) == 1:
        return str(parts[0])

    return None

#############################################################################################################################################
#################################### Get the most appropriate Crossref publication date #####################################################
def extract_crossref_publication_date(item):
    """
    Select the best available publication date from Crossref metadata.
    """
    date_fields = [
        "published-online",
        "published-print",
        "published",
        "issued",
        "created",
    ]

    for field in date_fields:
        date_value = crossref_date_to_string(
            item.get(field)
        )

        if date_value:
            return date_value

    return None
##############################################################################################################################################
###############################################Convert authors to readable names #############################################################
def extract_crossref_authors(item):
    """
    Extract author names from a Crossref work record.
    """
    authors = []

    for author in item.get("author", []):
        given = clean_whitespace(author.get("given"))
        family = clean_whitespace(author.get("family"))

        full_name = " ".join(
            part
            for part in [given, family]
            if part
        )

        if full_name:
            authors.append(full_name)

    return authors

################################################################################################################################################
#Search Crossref
def search_crossref(
    query,
    maximum_results=20,
    from_year=None,
    until_year=None,
):
    """
    Search Crossref for journal articles matching a query.

    Parameters
    ----------
    query : str
        Scientific search query.
    maximum_results : int
        Maximum records to request.
    from_year : int | None
        Optional beginning publication year.
    until_year : int | None
        Optional ending publication year.

    Returns
    -------
    list[dict]
        Normalized candidate-publication records.
    """
    endpoint = "https://api.crossref.org/works"

    filters = ["type:journal-article"]

    if from_year:
        filters.append(
            f"from-pub-date:{from_year}-01-01"
        )

    if until_year:
        filters.append(
            f"until-pub-date:{until_year}-12-31"
        )

    params = {
        "query.bibliographic": query,
        "rows": min(maximum_results, 100),
        "filter": ",".join(filters),
        "mailto": CROSSREF_EMAIL,
    }

    response = requests.get(
        endpoint,
        params=params,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()
    items = payload.get("message", {}).get("items", [])

    candidates = []

    for item in items:
        title_values = item.get("title") or []
        journal_values = item.get("container-title") or []

        title = (
            strip_html_tags(title_values[0])
            if title_values
            else None
        )

        if not title:
            continue

        journal = (
            strip_html_tags(journal_values[0])
            if journal_values
            else None
        )

        doi = normalize_doi(item.get("DOI"))

        article_url = (
            doi_to_url(doi)
            or item.get("URL")
        )

        abstract = strip_html_tags(
            item.get("abstract")
        )

        candidates.append({
            "title": title,
            "authors": extract_crossref_authors(item),
            "journal": journal,
            "publication_date": (
                extract_crossref_publication_date(item)
            ),
            "doi": doi,
            "article_url": article_url,
            "abstract": abstract,
            "source": "Crossref",
            "source_query": query,
            "publisher": clean_whitespace(
                item.get("publisher")
            ),
            "type": item.get("type"),
        })

    return candidates

######################################################################################################################################
#Search using all generated queries
def search_all_queries(
    search_queries,
    results_per_query=15,
    from_year=None,
    until_year=None,
    delay_seconds=0.5,
):
    """
    Search Crossref using every generated search query.
    """
    all_candidates = []
    failed_queries = []

    for query in search_queries:
        try:
            print(f"Searching: {query}")

            results = search_crossref(
                query=query,
                maximum_results=results_per_query,
                from_year=from_year,
                until_year=until_year,
            )

            all_candidates.extend(results)

        except requests.RequestException as error:
            failed_queries.append({
                "query": query,
                "error": str(error),
            })

        time.sleep(delay_seconds)

    return {
        "candidates": all_candidates,
        "failed_queries": failed_queries,
    }

#######################################################################################################################################
#Deduplicate publications
def normalize_title(title):
    """
    Create a normalized title for duplicate comparison.
    """
    if not title:
        return ""

    title = title.lower()
    title = re.sub(r"<[^>]+>", " ", title)
    title = re.sub(r"[^a-z0-9]+", " ", title)

    return re.sub(r"\s+", " ", title).strip()

#######################################################################################################################################
def deduplicate_publications(publications):
    """
    Remove duplicate publications.

    Priority:
    1. DOI
    2. Normalized title
    """
    unique_publications = []
    seen_dois = set()
    seen_titles = set()

    for publication in publications:
        doi = normalize_doi(publication.get("doi"))
        normalized_title = normalize_title(
            publication.get("title")
        )

        if doi and doi in seen_dois:
            continue

        if not doi and normalized_title in seen_titles:
            continue

        if doi:
            seen_dois.add(doi)

        if normalized_title:
            seen_titles.add(normalized_title)

        unique_publications.append(publication)

    return unique_publications

#######################################################################################################################################
##################################### Classify all candidates based on relevance and confidence score #################################
def filter_relevant_publications(
    topic,
    candidates,
    relevance_threshold=0.70,
    maximum_articles=None,
):
    """
    Classify candidate publications and separate relevant,
    rejected, and failed records.
    """
    relevant = []
    rejected = []
    failed = []

    publications_to_process = candidates

    if maximum_articles is not None:
        publications_to_process = candidates[
            :maximum_articles
        ]

    total = len(publications_to_process)

    for index, publication in enumerate(
        publications_to_process,
        start=1,
    ):
        print(
            f"Classifying {index}/{total}: "
            f"{publication['title']}"
        )

        try:
            relevance = classify_article_relevance(
                topic=topic,
                publication=publication,
            )

            assessed_publication = {
                **publication,
                "relevance": relevance,
            }

            if (
                relevance["is_relevant"]
                and relevance["relevance_score"]
                >= relevance_threshold
            ):
                relevant.append(assessed_publication)
            else:
                rejected.append(assessed_publication)

        except Exception as error:
            failed.append({
                **publication,
                "error": str(error),
            })

    return {
        "relevant": relevant,
        "rejected": rejected,
        "failed": failed,
    }
#################################################################################################################################################
###################################### Layer 2: Retrieve Publisher Article Pages ################################################################
#Download one webpage. This avoids downloading the same page separately for links and text
def fetch_html(url, timeout=30):
    """
    Download HTML and return retrieval information.
    Raises an exception for HTTP errors or non-HTML responses.
    """
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=timeout,
        allow_redirects=True,
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type","",
    ).lower()

    if (
        "text/html" not in content_type
        and "application/xhtml+xml" not in content_type
    ):
        raise ValueError(
            f"URL did not return HTML: {content_type}"
        )

    return {
        "requested_url": url,
        "final_url": response.url,
        "status_code": response.status_code,
        "html": response.text,
    }

###################################################################################################################################################
######################################## Layer 2: Extract Deterministic Article Metadata ##########################################################
#Read a tag
def get_meta_value(soup, names):
    """
    Return the first matching metadata value.
    """
    for name in names:
        tag = soup.find(
            "meta",
            attrs={"name": name},
        )

        if tag and tag.get("content"):
            return clean_whitespace(
                tag["content"]
            )

        tag = soup.find(
            "meta",
            attrs={"property": name},
        )

        if tag and tag.get("content"):
            return clean_whitespace(
                tag["content"]
            )

    return None

################################################################################################################################################
##################################################### Layer 2: Extract JSON-LD objects #########################################################
def extract_json_ld_objects(soup):
    """
    Extract valid JSON-LD objects from a webpage.
    """
    objects = []

    scripts = soup.find_all(
        "script",
        attrs={"type": "application/ld+json"},
    )

    for script in scripts:
        raw_json = script.string

        if not raw_json:
            continue

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            continue

        if isinstance(data, list):
            objects.extend(data)
        elif isinstance(data, dict):
            graph = data.get("@graph")

            if isinstance(graph, list):
                objects.extend(graph)
            else:
                objects.append(data)

    return objects

###################################################################################################################################################
############################################ Layer 2: Locate scholarly JSON-LD ####################################################################
def find_scholarly_json_ld(json_ld_objects):
    """
    Find an article-like JSON-LD object.
    """
    valid_types = {
        "ScholarlyArticle",
        "Article",
        "MedicalScholarlyArticle",
        "NewsArticle",
    }

    for item in json_ld_objects:
        item_type = item.get("@type")

        if isinstance(item_type, list):
            if valid_types.intersection(item_type):
                return item

        elif item_type in valid_types:
            return item

    return {}

##################################################################################################################################################
##################################################Layer 2: Normalize JSON-LD authors #############################################################
def extract_json_ld_authors(article_object):
    """
    Extract author names from an article JSON-LD object.
    """
    author_data = article_object.get("author", [])

    if isinstance(author_data, dict):
        author_data = [author_data]

    if isinstance(author_data, str):
        return [clean_whitespace(author_data)]

    authors = []

    for author in author_data:
        if isinstance(author, str):
            name = clean_whitespace(author)
        elif isinstance(author, dict):
            name = clean_whitespace(author.get("name"))
        else:
            name = None

        if name:
            authors.append(name)

    return authors

##################################################################################################################################################
########################################## Layer 2: Extract article metadata #####################################################################
def extract_article_metadata(
    html_text,
    final_url,
    discovery_record=None,
):
    """
    Extract article metadata from:
    1. citation meta tags;
    2. JSON-LD;
    3. discovery metadata as fallback.
    """
    discovery_record = discovery_record or {}

    soup = BeautifulSoup(
        html_text,
        "html.parser",
    )

    json_ld_objects = extract_json_ld_objects(soup)
    article_object = find_scholarly_json_ld(
        json_ld_objects
    )

    meta_title = get_meta_value(
        soup,
        [
            "citation_title",
            "dc.title",
            "DC.Title",
            "og:title",
        ],
    )

    page_title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else None
    )

    title = (
        meta_title
        or clean_whitespace(article_object.get("headline"))
        or discovery_record.get("title")
        or clean_whitespace(page_title)
    )

    journal = (
        get_meta_value(
            soup,
            [
                "citation_journal_title",
                "dc.source",
                "DC.Source",
            ],
        )
        or clean_whitespace(
            article_object.get(
                "isPartOf",
                {},
            ).get("name")
        )
        or discovery_record.get("journal")
    )

    publication_date = (
        get_meta_value(
            soup,
            [
                "citation_publication_date",
                "citation_date",
                "dc.date",
                "DC.Date",
                "article:published_time",
            ],
        )
        or clean_whitespace(
            article_object.get("datePublished")
        )
        or discovery_record.get("publication_date")
    )

    doi = (
        get_meta_value(
            soup,
            [
                "citation_doi",
                "dc.identifier",
                "DC.Identifier",
            ],
        )
        or article_object.get("identifier")
        or discovery_record.get("doi")
    )

    if isinstance(doi, dict):
        doi = doi.get("value")

    doi = normalize_doi(doi)

    abstract = (
        get_meta_value(
            soup,
            [
                "citation_abstract",
                "dc.description",
                "DC.Description",
            ],
        )
        or clean_whitespace(
            article_object.get("abstract")
        )
        or discovery_record.get("abstract")
    )

    meta_authors = [
        clean_whitespace(tag.get("content"))
        for tag in soup.find_all(
            "meta",
            attrs={"name": "citation_author"},
        )
        if clean_whitespace(tag.get("content"))
    ]

    authors = (
        meta_authors
        or extract_json_ld_authors(article_object)
        or discovery_record.get("authors", [])
    )

    canonical_tag = soup.find(
        "link",
        attrs={"rel": "canonical"},
    )

    canonical_url = (
        canonical_tag.get("href")
        if canonical_tag
        and canonical_tag.get("href")
        else final_url
    )

    canonical_url = urljoin(
        final_url,
        canonical_url,
    )

    return {
        "title": title,
        "authors": authors,
        "journal": journal,
        "publication_date": publication_date,
        "doi": doi,
        "article_url": canonical_url,
        "abstract": clean_whitespace(abstract),
    }

########################################################################################################################################################
##################################################### Layer 2: Extract Useful Article Text ######################################################################
def remove_unwanted_elements(soup):
    """
    Remove common webpage boilerplate.
    """
    unwanted_tags = [
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "form",
        "button",
        "input",
        "nav",
        "footer",
        "header",
        "aside",
    ]

    for element in soup.find_all(unwanted_tags):
        element.decompose()

    unwanted_attributes = [
        {"role": "navigation"},
        {"role": "banner"},
        {"role": "contentinfo"},
    ]

    for attributes in unwanted_attributes:
        for element in soup.find_all(attrs=attributes):
            element.decompose()

    return soup

######################################################################################################################################################
def extract_article_text(
    html_text,
    maximum_characters=60_000,
):
    """
    Extract the most useful visible text from an article page.

    Prefers:
    1. <article>
    2. <main>
    3. the document body
    """
    soup = BeautifulSoup(
        html_text,
        "html.parser",
    )

    soup = remove_unwanted_elements(soup)

    content = (
        soup.find("article")
        or soup.find("main")
        or soup.body
        or soup
    )

    text = content.get_text(
        separator="\n",
        strip=True,
    )

    text = clean_whitespace(text) or ""

    return text[:maximum_characters]

########################################################################################################################################################
####################################################### Layer 2: Determine Access Level ################################################################
def determine_access_level(metadata, article_text):
    """
    Estimate how much article content was retrieved.
    """
    text_length = len(article_text or "")
    has_abstract = bool(metadata.get("abstract"))

    full_text_markers = [
        "results",
        "discussion",
        "materials and methods",
        "methods",
        "conclusion",
    ]

    lowered_text = (article_text or "").lower()

    marker_count = sum(
        marker in lowered_text
        for marker in full_text_markers
    )

    if text_length > 15_000 and marker_count >= 2:
        return "full_text"

    if has_abstract:
        return "abstract_only"

    if text_length > 500:
        return "metadata_and_page_text"

    return "metadata_only"

##########################################################################################################################################################
################################################# Layer 3:  Basic Output Validation ######################################################################
def validate_publication_record(record):
    """
    Apply basic validation to an extracted publication record.
    """
    errors = []
    warnings = []

    if not record.get("title"):
        errors.append("Missing publication title.")

    if not record.get("article_url"):
        errors.append("Missing article URL.")

    doi = record.get("doi")

    if doi and not re.match(
        r"^10\.\d{4,9}/\S+$",
        doi,
        flags=re.IGNORECASE,
    ):
        warnings.append(
            f"DOI format may be invalid: {doi}"
        )

    findings = record.get("main_findings", [])

    for index, finding in enumerate(
        findings,
        start=1,
    ):
        if not finding.get("finding"):
            errors.append(
                f"Finding {index} has no finding text."
            )

        if not finding.get("evidence"):
            warnings.append(
                f"Finding {index} has no evidence excerpt."
            )

    record["validation"] = {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }

    return record
##############################################################################################################################################
#################################################### ### Publication Extraction With Model 2 #################################################
def extract_publication_information(
    metadata,
    article_text,
    access_level,
):
    """
    Use Model 2 to extract the study objective, methods,
    main findings, and data resources.
    """
    system_prompt = """
You are a scientific publication information-extraction assistant with
expertise in genomics, genetics, sequencing, bioinformatics, molecular
biology, and biomedical research.

Use only the supplied metadata and article content.

Rules:
1. Do not use outside knowledge.
2. Do not invent missing values.
3. Preserve deterministic metadata when provided.
4. Do not rewrite the abstract. Return the supplied abstract exactly.
5. Extract only findings supported by the supplied content.
6. Do not convert associations or correlations into causal claims.
7. If only an abstract is available, report only findings stated in
   the abstract.
8. For every main finding, provide a short supporting evidence excerpt.
9. Do not claim to have read the full publication when the access level
   is abstract_only or metadata_only.
10. Use null or empty lists for unavailable information.
11. Return valid JSON only.
12. Do not include Markdown or commentary.

Return this structure:

{
  "title": "title",
  "authors": ["author"],
  "journal": "journal or null",
  "publication_date": "date or null",
  "doi": "DOI or null",
  "article_url": "URL",
  "abstract": "exact supplied abstract or null",
  "study_objective": "objective or null",
  "main_findings": [
    {
      "finding": "concise finding",
      "evidence": "short supporting excerpt",
      "source_section": "abstract, results, discussion, or null"
    }
  ],
  "methods": [],
  "sequencing_technologies": [],
  "variant_analysis_methods": [],
  "data_resources": [
    {
      "repository": "repository name",
      "accession": "accession or null",
      "url": "URL or null"
    }
  ],
  "limitations_of_extraction": [],
  "access_level": "provided access level"
}
"""

    user_prompt = f"""
Access level:
{access_level}

Deterministic metadata:
{json.dumps(metadata, indent=2, ensure_ascii=False)}

Extracted article content:
--- BEGIN ARTICLE CONTENT ---
{article_text}
--- END ARTICLE CONTENT ---
"""

    raw_response = call_chat_model(
        client=openrouter,
        model=PUBLICATION_EXTRACTION_MODEL["model"],
        temperature=PUBLICATION_EXTRACTION_MODEL["temperature"],
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    extracted = parse_json_response(raw_response)

    # Deterministic metadata takes precedence over model output.
    extracted["title"] = metadata.get("title")
    extracted["authors"] = metadata.get("authors", [])
    extracted["journal"] = metadata.get("journal")
    extracted["publication_date"] = metadata.get(
        "publication_date"
    )
    extracted["doi"] = metadata.get("doi")
    extracted["article_url"] = metadata.get(
        "article_url"
    )
    extracted["abstract"] = metadata.get("abstract")
    extracted["access_level"] = access_level

    extracted.setdefault("main_findings", [])
    extracted.setdefault("methods", [])
    extracted.setdefault(
        "sequencing_technologies",
        [],
    )
    extracted.setdefault(
        "variant_analysis_methods",
        [],
    )
    extracted.setdefault("data_resources", [])
    extracted.setdefault(
        "limitations_of_extraction",
        [],
    )

    return extracted

############################################################################################################################################
############################################# Layer 3: Process one article #################################################################
def process_article(publication):
    """
    Retrieve and extract one relevant publication.
    """
    article_url = publication.get("article_url")

    if not article_url:
        raise ValueError(
            "Publication does not have an article URL."
        )

    retrieval = fetch_html(article_url)

    metadata = extract_article_metadata(
        html_text=retrieval["html"],
        final_url=retrieval["final_url"],
        discovery_record=publication,
    )

    article_text = extract_article_text(
        html_text=retrieval["html"],
        maximum_characters=60_000,
    )

    access_level = determine_access_level(
        metadata=metadata,
        article_text=article_text,
    )

    extracted = extract_publication_information(
        metadata=metadata,
        article_text=article_text,
        access_level=access_level,
    )

    validated = validate_publication_record(
        extracted
    )

    return validated

####################################################################################################################################################
######################################### Layer 3: Process all relevant articles ###################################################################
def process_relevant_publications(
    publications,
    delay_seconds=1.0,
):
    """
    Process relevant publications independently so that one
    failure does not stop the full pipeline.
    """
    successful = []
    failed = []

    total = len(publications)

    for index, publication in enumerate(
        publications,
        start=1,
    ):
        print(
            f"Extracting {index}/{total}: "
            f"{publication['title']}"
        )

        try:
            record = process_article(publication)
            successful.append(record)

        except Exception as error:
            failed.append({
                "title": publication.get("title"),
                "article_url": publication.get(
                    "article_url"
                ),
                "doi": publication.get("doi"),
                "error": str(error),
            })

        time.sleep(delay_seconds)

    return {
        "successful": successful,
        "failed": failed,
    }

#####################################################################################################################################################
############################################## Layer 4: Save JSON ###################################################################################
def save_json(data, output_path):
    """
    Save any Python dictionary or list as JSON.
    """
    output_file = Path(output_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return str(output_file)
    
##########################################################################################################################

