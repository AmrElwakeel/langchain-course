from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


load_dotenv()


def main():
    information = """
    Sir Magdi Habib Yacoub[a] (born 16 November 1935) is a Coptic Egyptian-British[1] retired professor of cardiothoracic surgery at Imperial College London, best known for his early work in repairing heart valves with surgeon Donald Ross, adapting the Ross procedure, where the diseased aortic valve is replaced with the person's own pulmonary valve, devising the arterial switch operation (ASO) in transposition of the great arteries, and establishing the heart transplantation centre at Harefield Hospital in 1980 with a heart transplant for Derrick Morris, who at the time of his death was Europe's longest-surviving heart transplant recipient. Yacoub subsequently performed the UK's first combined heart and lung transplant in 1983.

    From 1986 to 2006, he held the position of British Heart Foundation Professor of Cardiothoracic Surgery at the National Heart and Lung Institute, Imperial College Faculty of Medicine. He is the founding editor of the journal Disease Models & Mechanisms.

    His honours and awards include the Bradshaw Lecture from the Royal College of Physicians in 1988, a knighthood in the 1992 New Year Honours, the Texas Heart Institute's Ray C. Fish Award for Scientific Achievement in Cardiovascular Disease in 1998, the International Society for Heart and Lung Transplantation Lifetime Achievement Award in 2004, the European Society of Cardiology's gold medal in 2006, the Order of Merit in 2014, and the Lister Medal from the Royal College of Surgeons in 2015.

    After retiring from the National Health Service (NHS), he continued to practice paediatric surgery through his charity, Chain of Hope. In 2008, he co-founded the Magdi Yacoub Heart Foundation, which initiated the Aswan Heart Project.
    """
    
    summary_template = """
    Based on the following information {information}, give me below information:
    1- a simple summary about him.
    2- the most two information about him
    """

    prompt = PromptTemplate(template=summary_template, input_variables=["information"])

    chain = prompt | ChatOllama(model="qwen3.6:latest")

    result = chain.invoke({"information": information})

    print(result.content)

if __name__ == "__main__":
    main()
