from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document


document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}"
)
document_variable_name = "context"
llm = ChatOpenAI()

prompt = PromptTemplate.from_template(
    """Use the following pieces of context to answer the question at the end: {context},
    question: {question}"""
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
    verbose=True,
)
input = {'input_documents': 
         [Document(page_content='5.2.6新员工转为正式员工后由人事行政部以邮件形式发出《转正通知单》的形式以通知转正人员相关\n事宜。6.转正类别转正可分为提前转正、按期转正、终止试用四个类别。\n6.1提前转正：新员工入职三个月以后，确因工作态度、工作绩效、团队融合表现突出者，部门负责人\n可为员工申请提前转正。并依据本办法5.2试用期满评估程序实施转正。\n6.2按期转正：依据本办法5.2试用期满评估程序实施转正。\n6.3终止试用\n6.3.1试用期员工在试用期如不能认同企业文化，感到公司状况与个人预期差距较大或者其它原因决定\n离开公司的，可提出辞职，但应提前3日以书面形式提交辞职报告。\n6.3.2试用期员工在试用期间如不能达到公司岗位任职要求，公司可随时停止试用，予以辞退。\n6.3.3试用期员工在试用期间有严重违反公司规章制度行为并造成不良影响的，公司可随时停止试用，\n予以辞退。\n6.3.4终止试用的员工，至人事行政部办理离职手续。7.转正日期新员工如果提前转正，则转正日期以最终审批的提前转正日期为准；如果按期转正，新员工的转正日期\n为试用期满之日。8.其他及附件8.1本办法自发布之日起生效。\n8.2附件\n《月度工作计划表》\n《试用期转正述职报告》\n《转正述职与答辩评分表》\n《试用期转正评估表》\n《员工转正通知书》\n二○一九年八月[COMPANY_NAME_CN]', metadata={'source': '试用期评估及转正管理办法.pdf', 'page': 1, 'test': 'test'}),
          Document(page_content='试用期评估及转正管理办法\n1目的\n为帮助员工发展、统一员工在试用期内评估及转正要求与流程，明确各相关部门的工作职责，特制定本\n办法。\n2适用范围\n本办法适用于公司所有新进员工。\n3定义\n3.1试用期：指劳动合同期限内公司与员工为相互了解对方而约定的考察期间。\n3.2试用期期限根据相关劳动法律法规及工作岗位性质确定。\n3.3转正：指新员工试用期满，达到岗位任职资格，并按规定办完相应手续后，成为公司的正式员工。\n4各方权责\n4.1试用期员工的权责\n主动了解自己的工作职责与工作内容、工作目标、公司文化等各项规章制度；\n接受人力部门的入职培训与相关评估；\n4.2试用期各部门负责人的权责\n在工作中引导新员工领悟并融入公司的文化、督促并指导其执行公司各项任务、工作须知及管理制度；\n帮助新员工，使其了解并掌握所任职岗位的岗位职责、内容、目标与岗位应知应会；\n4.3人事行政部的权责\n培训新员工有关公司的组织、文化、发展、管理制度及日常办公工作须知等；\n负责试用期员工的评估执行。\n5试用期评估\n试用期员工的评估分为两种形式：中间评估与期满综合评估。中间评估自入职之日起试用期内分阶段（每\n阶段按月或者按照季度设定）进行，期满综合评估在试用期结束时进行。\n5.1中间评估程序\n5.1.1部门负责人必须在入职第一个月内与新员工确定阶段性（月度或者季度）工作目标。\n5.1.2在阶段结束时，部门负责人与新员工对本阶段学习任务与工作目标达成情况以及下阶段工作计划\n与目标进行沟通，对新员工遇到的疑问进行合理解答，对新员工遇到的困难进行及时解决。\n5.1.3中间评估完成后由部门负责人批准将表格原件备案至人事行政部，人事行政部对月度评估表的合\n理性进行评估确认。\n5.2试用期满综合评估\n5.2.1试用期期满前10天，人事行政部将试用期《试用期转正述职报告模板》发给新员工，抄送部门\n负责人；将《试用期转正评估表》发给部门负责人；\n5.2.2新员工填写《试用期转正述职报告模板》，对试用期间的学习与工作心得进行总结，于转正期满\n前一周完成并交部门负责人审核，交人事行政部备案；\n5.2.3部门负责人初步考核通过后，人事行政部安排新员工转正述职，邀请关联团队经理、部门负责人、\n公司负责人、HR作为评审人员参与转正述职评审；\n5.2.4评审人员根据新员工现场述职、述职材料、既往的中间评估给予综合评分，最终评分≥3.5分方\n为通过转正述职评估；\n5.2.5部门负责人需于新员工试用期满前3天完成《试用期转正评估表》；', metadata={'source': '试用期评估及转正管理办法.pdf', 'page': 0})
          ]}
input["question"] = "新员工怎么转正"
# result = chain(input)
result = chain.invoke(input)
print(result.keys())  # dict_keys(['input_documents', 'question', 'output_text'])
# print(result)
print(result["output_text"])