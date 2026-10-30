"""Build the four-page portfolio PDF from archived CSVs. Requires reportlab."""
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from analyze_results import ROOT, RUNS, read_run, summarize

OUT = ROOT / "output/pdf/robot_imitation_report.pdf"
W, H = 595.28, 841.89
INK = colors.HexColor("#10233F")
MUTED = colors.HexColor("#52647B")
BLUE = colors.HexColor("#2563EB")
GREEN = colors.HexColor("#059669")
LIGHT = colors.HexColor("#EFF4FA")
STYLE = ParagraphStyle("body", fontName="Helvetica", fontSize=11.2, leading=16.4, textColor=INK)
SMALL = ParagraphStyle("small", parent=STYLE, fontSize=9, leading=12.5, textColor=MUTED)


def text(c, value, x=44, y=700, width=507, small=False):
    p = Paragraph(value, SMALL if small else STYLE)
    _, height = p.wrap(width, 900)
    if y - height < 56:
        raise ValueError(f"Page overflow near: {value[:75]}")
    p.drawOn(c, x, y - height)
    return y - height - 12


def heading(c, title, y):
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(44, y, title)
    return y - 18


def page(c, n, tag):
    c.setFillColor(colors.white)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.rect(44, H - 43, 30, 4, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(84, H - 43, "WATCH, PLAN, ACT")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawRightString(W - 44, H - 43, tag.upper())
    c.setStrokeColor(colors.HexColor("#D8E2EE"))
    c.line(44, 44, W - 44, 44)
    c.setFont("Helvetica", 8)
    c.drawString(44, 29, "Chao-Hsu Chen  |  CSC2626 course-derived robotics project  |  Simulation")
    c.drawRightString(W - 44, 29, f"{n} / 4")


def title(c, words, subtitle):
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 29)
    c.drawString(44, 747, words)
    return text(c, subtitle, y=725, small=True)


def box(c, x, y, w, h, title_, body, color=LIGHT):
    c.setFillColor(color)
    c.roundRect(x, y-h, w, h, 10, fill=1, stroke=0)
    st = ParagraphStyle("box", parent=SMALL, textColor=INK, fontSize=10, leading=14)
    p = Paragraph(f"<b>{title_}</b><br/>{body}", st)
    _, ph = p.wrap(w-24, h)
    if ph > h-18:
        raise ValueError("Box overflow")
    p.drawOn(c, x+12, y-12-ph)


def arrow(c, x1, y1, x2, y2):
    c.setStrokeColor(MUTED)
    c.setFillColor(MUTED)
    c.setLineWidth(1.3)
    c.line(x1,y1,x2,y2)
    p=c.beginPath();p.moveTo(x2,y2);p.lineTo(x2-5,y2+3);p.lineTo(x2-5,y2-3);p.close()
    c.drawPath(p,fill=1,stroke=0)


def chart(c, names, x, y, width=507, height=205):
    left=x+32; bottom=y-height+30; pw=width-42; ph=height-45
    c.setFont("Helvetica",8)
    for v in [0,.25,.5,.75,1]:
        yy=bottom+v*ph
        c.setStrokeColor(colors.HexColor("#E2E8F0")); c.line(left, yy, left+pw, yy)
        c.setFillColor(MUTED);c.drawRightString(left-7, yy-3,f"{v*100:.0f}%")
    for s in [20,40,60,80,100]:
        c.drawCentredString(left+(s-10)/90*pw,bottom-14,f"{s}k")
    for i,name in enumerate(names):
        label,col=RUNS[name]; color=colors.HexColor(col)
        rows=read_run(name)
        c.setStrokeColor(color); c.setLineWidth(2)
        path=c.beginPath()
        for j,(s,r) in enumerate(rows):
            xx=left+(s/1000-10)/90*pw; yy=bottom+r*ph
            if j==0:path.moveTo(xx,yy)
            else:path.lineTo(xx,yy)
        c.drawPath(path)
        c.setFillColor(color)
        for s,r in rows:c.circle(left+(s/1000-10)/90*pw,bottom+r*ph,2.4,fill=1,stroke=0)
        c.setFont("Helvetica",8)
        c.drawString(left+i*158,y+5,label.replace("epsilon","eps"))
    c.setFillColor(MUTED);c.drawCentredString(left+pw/2,bottom-27,"Training updates")


def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(OUT),pagesize=(W,H))
    c.setTitle("Watch, Plan, Act: Diffusion Policies for Robot Imitation Learning")
    c.setAuthor("Chao-Hsu Chen")
    c.setSubject("Conversational project report: architecture, evaluation, and findings")
    metrics=summarize()

    page(c,1,"The problem")
    y=title(c,"Watch, Plan, Act", "Diffusion Policies for Robot Imitation Learning | A guided project report")
    y=text(c,"<b>Can a robot learn a precise manipulation task from human examples?</b> That is the question behind this project. A simulated Franka Panda must grasp a square nut, align it, and fit it onto a peg. It sees the scene through cameras and decides how to move its gripper.",y=y-14)
    box(c,44,y,507,82,"The result at a glance","At 100,000 updates: <b>20% success for single-action BC, 48% for diffusion noise prediction, and 52% for diffusion clean-action prediction.</b> Each score comes from 25 simulator episodes.")
    y-=103
    y=heading(c,"Why learn from demonstrations?",y)
    y=text(c,"Rather than hand-code every grasp and alignment correction, we show the model what a human did in each situation. It learns a mapping from the current observation to a robot command. This is imitation learning: the training targets come from demonstrations, not from optimizing evaluation rewards.",y=y)
    y=heading(c,"What is in one training example?",y-5)
    box(c,44,y,244,111,"What the robot sees","Two 84 x 84 RGB images: external and wrist views.<br/>End-effector position and quaternion orientation.<br/>Gripper position.")
    box(c,303,y,248,111,"What the human commanded","A 7-dimensional action:<br/>3 position increments<br/>3 rotation increments (axis-angle)<br/>1 gripper command")
    y-=133
    y=text(c,"The local Robomimic square/ph/image dataset contains <b>200 proficient-human demonstrations and 30,154 time steps</b>. Only the selected camera and robot-state fields are used. The policy does not receive the stored ground-truth object state.",y=y)
    y=text(c,"<b>Project context.</b> This work is based on CSC2626: Imitation Learning for Robotics. The course supplied the network and simulator scaffolding. The implemented algorithm components include DDPM noising, both prediction objectives, reverse sampling, and action chunk execution. This report explains the resulting system and analyzes its saved evaluations.",y=y)
    text(c,"All robot experiments discussed here use simulation. No physical robot deployment is claimed.",y=y,small=True)
    c.showPage()

    page(c,2,"The architecture")
    y=title(c,"One view, two policies", "Both methods use the same observation and backbone architecture, with separate learned weights.")
    box(c,44,681,150,89,"Observe","External + wrist images<br/>End-effector pose<br/>Gripper position")
    box(c,222,681,148,89,"Encode","ResNet18 image features<br/>+ robot state<br/>256-d condition vector")
    box(c,398,681,153,89,"Choose actions","DiT1d backbone<br/>BC: one action<br/>Diffusion: short plan")
    arrow(c,196,637,218,637);arrow(c,372,637,394,637)
    y=heading(c,"BC asks: what would the human do now?",563)
    y=text(c,"Behavior cloning directly predicts one action and minimizes mean squared error against the demonstrated action. It executes that command, looks again, and repeats. Here the DiT action input and timestep are fixed to zero. The observation condition still changes, so the model can respond to the scene without any diffusion process.",y=y)
    y=text(c,"It is called BC because of <b>how it learns: supervised imitation of demonstrated behavior</b>. Reusing a Transformer architecture does not make it diffusion. In the broader sense, the diffusion policies also learn by imitation; BC names the direct-regression baseline in this comparison.",y=y)
    y=heading(c,"Diffusion asks: what short plan fits this view?",y-4)
    y=text(c,"During training, add Gaussian noise to a demonstrated 10-action sequence. The network sees the noisy sequence, its noise level, and the observation condition. The epsilon variant learns to predict the added noise; the x0 variant learns to predict the original clean sequence.",y=y)
    box(c,44,y,507,69,"At execution time","Start with random noise -> apply 50 denoising steps -> generate 10 actions -> execute the first 8 -> observe again and replan.",colors.HexColor("#DBEAFE"))
    y-=88
    y=text(c,"Those 50 steps happen inside the model's sampling loop; they are not 50 robot moves. Executing eight planned actions before replanning is <b>action chunking</b>. The architecture aims to represent coherent short motion plans and multiple possible actions, but the measured success rates alone do not prove those mechanisms caused the improvement.",y=y)
    text(c,"Implementation map: Q3/bc.py; Q3/ddpm.py; Q3/diffusion_policy.py; Q3/networks/; Q3/utils/dataset.py. Backbone: width 320, two blocks, ten attention heads. Predicted actions are unnormalized before simulator execution.",y=y,small=True)
    c.showPage()

    page(c,3,"The findings")
    y=title(c,"What did the robot learn?", "Main experiment | Success rate over training | 25 evaluation episodes per checkpoint")
    chart(c,list(RUNS)[:3],44,675)
    y=442
    labels={"bc":"BC", "diffusion_policy_eps":"Diffusion eps / cosine", "diffusion_policy_x0":"Diffusion x0 / cosine", "diffusion_policy_eps_sigmoid":"Bonus: eps / sigmoid"}
    for i,(name,m) in enumerate(metrics.items()):
        yy=y-i*30
        if i%2==0:
            c.setFillColor(LIGHT);c.rect(44,yy-9,507,29,fill=1,stroke=0)
        c.setFillColor(INK);c.setFont("Helvetica",10)
        c.drawString(55,yy,labels[name]);c.drawString(260,yy,f"Final: {m['final_success']:.0%}")
        c.drawString(360,yy,f"Peak: {m['best_success']:.0%} at {m['best_step']//1000}k")
    y=311
    y=text(c,"<b>The diffusion-plus-chunking setup finished ahead of single-action BC.</b> BC ended at 20%, versus 48% and 52% for the two main diffusion variants. This supports the combined design in this run; it does not separate the benefit of diffusion from the benefit of action chunking.",y=y)
    y=text(c,"<b>Clean-action prediction peaked higher, then fell.</b> The x0 model reached 88% at 80k updates and finished at 52%. The observed curve is not monotonic. That drop alone does not tell us whether the cause is training instability, evaluation variation, or another mechanism.",y=y)
    y=text(c,"<b>The noise schedule mattered in the bonus run.</b> Changing epsilon prediction from cosine to sigmoid ended at 60% rather than 48%, with a peak of 76%. This is a promising setup-specific observation, not a general ranking of schedules.",y=y)
    text(c,"Configuration from code/script defaults: 100k updates, batch 64, learning rate 1e-4, seed 0, evaluation every 10k, 500-step limit. CSVs record scores and update counts, but do not contain a complete original run manifest. Source: Q3/results/*/eval_metrics.csv.",y=y,small=True)
    c.showPage()

    page(c,4,"Interpretation and experience")
    y=title(c,"What can we claim?", "A useful robotics result is one we can explain, reproduce, and put in context.")
    y=heading(c,"Read the results with the right confidence",y-10)
    y=text(c,"There is <b>one recorded run per method</b>, with only 25 episodes at each checkpoint. One success changes the score by four percentage points. The final 52% versus 48% gap is just one outcome, so it does not establish that x0 reliably beats epsilon prediction. Peaks were selected from the same evaluation curve and need independent confirmation.",y=y)
    y=text(c,"The loader uses all 200 demonstrations, including the dataset's built-in validation subset. Evaluation is simulator execution, not held-out demonstration prediction. BC sums rewards at every step; diffusion sums a maximum within each action chunk. That is why this report compares <b>task success</b>, not raw accumulated rewards.",y=y)
    y=text(c,"The included videos visualize action-vector denoising, not successful robot rollouts. No failure-category analysis, hardware test, or policy latency benchmark is claimed. An equal number of training updates also does not imply an equal compute budget.",y=y)
    y=heading(c,"What practical experience does this show?",y-3)
    y=text(c,"The work connects visual learning to continuous robot control: preparing demonstrations, encoding two camera views, implementing DDPM objectives and sampling, executing action chunks, integrating MuJoCo evaluation, handling checkpoints, and running GPU jobs. The analysis makes the comparison auditable by retaining evaluation CSVs and deriving the figures from them.",y=y)
    y=heading(c,"What would I try next?",y-3)
    y=text(c,"First, train a chunked BC baseline with the same 10-action horizon and eight-action execution interval. Then repeat all methods across several seeds with larger, separate evaluation sets. Record task rollouts to distinguish grasp failures from alignment failures, and measure inference time. That would tell us more about why performance changes and what still prevents reliable operation.",y=y)
    y=heading(c,"Explore the project",y-3)
    y=text(c,'<link href="https://github.com/chaohsuchen/Diffusion-Policies-for-Robot-Imitation-Learning" color="#2563EB">GitHub: Diffusion Policies for Robot Imitation Learning</link><br/><link href="https://github.com/chaohsuchen/DDPM-From-Scratch" color="#2563EB">Companion project: DDPM math and a short demo</link>',y=y)
    text(c,"Evidence: Q3/results/{bc,diffusion_policy_eps,diffusion_policy_x0,diffusion_policy_eps_sigmoid}/eval_metrics.csv. Dataset counts: inspected local image.hdf5. Rebuild: scripts/analyze_results.py and scripts/build_report.py. Credits: CSC2626 scaffold; Diffusion Policy (Chi et al.); DDPM (Ho et al., 2020); Robomimic; Robosuite/MuJoCo; CleanDiffuser. Full provenance and reproduction notes are in docs/ATTRIBUTION.md and docs/REPRODUCING.md.",y=y,small=True)
    c.save()
    print(OUT)


if __name__=="__main__":
    main()
