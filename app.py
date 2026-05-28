import chainlit
from local_utilities import *
from local_utilities.dataset import tokenize_text
from local_utilities.gpu import get_gpu
import numpy as np
from pathlib import Path
import scipy
import subprocess

# Get torch device
device = get_gpu()

@chainlit.on_message
async def on_message(message: chainlit.Message):
    # Check if the text was input into the text field
    text_to_check = ""
    pdfpath = None
    received_message = None
    
    if len(message.elements) == 0:
        # Just message content -> use this
        text_to_check = message.content
    elif len(message.elements) == 1 and len(message.content) > 0:
        # Message content and PDF attached -> show error, we can only process one document in one message
        await chainlit.Message(
            content="You either attach a PDF or submit a text to check but not both at the same time."
        ).send()
        return
    elif len(message.elements) == 1:
        # One PDF attached -> use this
        pdfpath = message.elements[0].path
        received_message = chainlit.Message(content="Received PDF")
        await received_message.send()
        
        # PDF to TXT
        jarfile = "./PDFMining/build/libs/PDFExtract-1.0-SNAPSHOT.jar"
        jarpath = Path(jarfile)
        if not jarpath.exists():
            received_message.content = "The PDF extraction tool was not found. Did you build it?"
            await received_message.update()
            return
        
        received_message.content = "Preparing PDF text mining"
        await received_message.update()
        
        txtpath = f"{pdfpath}.txt"
        mining = subprocess.run(["java", "-jar", jarfile, f"--pdf={pdfpath}", f"--txt={txtpath}"], capture_output=True)
        print(mining)
        if len(mining.stderr) > 0:
            await chainlit.Message(
                content=f"Errors occurred during PDF mining: {str(mining.stderr)}"
            ).send()
        
        # Read TXT
        text_to_check = read_text_from_file(txtpath)                
        received_message.content = "Mined PDF file"
        await received_message.update()
    else:
        # Multiple PDFs attached -> show error, only one PDF can be processed per message
        await chainlit.Message(
            content="You can only upload one file per message."
        ).send()
        return
    
    # text_to_check now contains the text to check for
    if len(text_to_check) == 0:
        await chainlit.Message(
            content="The provided message or PDF file contains no text to evaluate"
        ).send()
        return
    
    if received_message:
        await received_message.remove()
    
    test_message = await chainlit.Message(
        content="Running tests, this will take some seconds"
    ).send()
        
    # Tokenize text
    result_text = ""
    texts = tokenize_text(text_to_check)
    
    # Run consecutive sentences text
    token, ffnn_results, distilbert_results, roberta_results, svm_results, gbm_probas = \
        evaluate(texts, device=device)
    htmlpath = f"{pdfpath}.html" if pdfpath else "/tmp/test.html"
    detections = generate_report(0.95, token, ffnn_results, distilbert_results, roberta_results, svm_results, gbm_probas, htmlpath)

    elements = [
        chainlit.File(
            name="consecutive_sentences_test_report.html",
            path=htmlpath,
            display="inline",
            mime="text/html",
        ),
    ]


    # Run gauss test if there are at least 15 sentences
    z_score = 0
    p_value_right = 0
    gauss_ran = False
    if len(texts) >= 10:
        gauss_ran = True
        
        overall_mean = np.float64(0.09416525027037424)
        std = np.float64(0.06914971730529827)
        CUTOFF = 0.5
        p_value_min = 0.001

        result = ((gbm_probas.T[1] > CUTOFF) == True).sum() / len(gbm_probas.T[1])
        z_score = (result - overall_mean) / std
        p_value_right = scipy.stats.norm.sf(z_score)
    else:
        result_text +=f"⛔ Gauss test is not applicable, because only {len(texts)} sentences were found. At least 10 sentences are required to run the Gauss test.\n"

        
    # Add table of results
    result_text += f"""<h3>Test results</h3>
    
| Test | Passed | Details|
|------|--------|--------|
| Consecutive sentences test | {"❌" if detections > 0 else "✅"} | The text contains {detections} consecutive sentences which are assumed to be generative AI fabricated. |\n"""

    if gauss_ran:
        result_text += f"""| Gauss test | {"❌" if p_value_right < p_value_min else "✅"} | The text is {z_score:.2f} standard deviations away from how humans write, leading to a p-value of {p_value_right:.4f}.|"""

    test_message.content = result_text
    await test_message.update()
    
    await chainlit.Message(
        content="📜 You can find the consecutive sentences report here:", elements=elements
    ).send()
