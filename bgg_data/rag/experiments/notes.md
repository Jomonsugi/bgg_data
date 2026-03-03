## ColiPali

With only a basic implementation of ColiPali, it appears to focus on text more than images in board game
instruction booklets. For example if the word "bag" is queried the max similarity is on the text "bag"on a png instead of an image of a bag. That might always suffice and would only fall short if a query needed to find an image where no text is present. Beacuse the images in board game instructions are almost always complementery to the text, this might not be valuable. And in turn, in terms of finding relevent pages in an instruction booklet, the text alone could be used and then finally the entire png/pdf could be leveraged by a VLM to ask a question where the compliementary image is useful to answering a quesetion about the rules. 

So for game rulebooks, when searching for relevant documents from a query, the text alone might suffice. Then a VLM could finally explain the rule based on the full image of the retrieved pages. Thus a more simple and less memory intensive vectorization strategy from llama index that specializes in text recognition on unstructured pdfs like board game instructions would suffice. 

Even with all rulebooks of interest, original embedding sizes will likely be of no concern for memory, however if that issues arises, various optimization techniques could be tested like scalar quantization, pooling, or hierarchical techniques.  

# Docling

Docling is able to extract components from the gameboard rulebook. It correctly partitions text and images and then outputs them into an object that can be leveraged by an LLM. The object was also human readable and easy to understand. Not only is the full text extracted, but each partition's contents are stored in what is called a bbox. This allows for citation and even highlighting of the source of LLM output. 

After the extration, with minimal tuning, as I hypothesized, simple text based retrieval was more performant than ColiPali or a hybrid approach. For a rulebook, it is difficult to find a situation where the text is not sufficient to find a page and in almost all cases answer a question. The images are almost always complimentary. That means that if the rule can be clarified and a correct citation made with easy navigation to the rulebook page, the image there could provide supporting context to a user without the LLM using the image. At least that is the simplest and possibly full sufficient route. 