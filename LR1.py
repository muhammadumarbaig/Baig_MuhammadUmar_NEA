from manim import*
from bs4 import BeautifulSoup
import numpy as np
import math
class SVG_Handler:
    def __init__(self, filename):
        self.filename = filename
        self.sobj = SVGMobject(self.filename)
    @staticmethod
    def id_to_index(xmlid, lib):#Iterator for a list of dictionaries style format.....
        #grabs index of the dict where the id inside the dict is equal to the one
        for i, path in enumerate(lib):
                if path["id"] == xmlid:
                    return i
           
    def g_id2c(self, g_id, mode="onlyindex"):
        with open(self.filename, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "xml")
            groupdict = soup.find_all("g", id=True)#List of dictionaries of those under group tag
            lib = soup.find_all(["path", "circle", "rect", "polygon", "polyline", "line", "ellipse"])#List of dictionaries of everything
            #Parsing through the groupdict, and extracting the index of the group where the id matches 
            for i, group in enumerate(groupdict):
                if group["id"] == g_id:
                    groupindex = i
                    break       
            subdict = groupdict[groupindex].find_all(["path", "circle", "rect", "polygon", "polyline", "line", "ellipse"], id=True)#We can only index 
            #into the list using an index, this index was found earlier  
            #Hence pathdict is the list of dictionaries for the paths(subcomponents) of the particular group ONLY
            if mode == "fulldict":
                for path in subdict:
                    pathid = path["id"]
                    
                    path["index"] = self.id_to_index(pathid, lib)
                return subdict
            
            elif mode == "idandindex":
                idandno = []
                for path in subdict:
                    pathid = path["id"]
                    index = self.id_to_index(pathid, lib)
                    idandno.append({"index": index, "id": pathid})
                return idandno

            elif mode == "onlyindex":
                onlyindex = []
                for path in subdict:
                    pathid = path["id"]
                    index = self.id_to_index(pathid, lib)
                    onlyindex.append(index)
                return onlyindex
    def CustomVGroup(self,svg, g_id):
        group = VGroup()
        indices = self.g_id2c(g_id, mode="onlyindex")
        for index in indices:
            group.add(svg[index])
        return group
            
    def conv_id(self, xmlid):
        with open(self.filename, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "xml")
            lib = soup.find_all(["path", "circle", "rect", "polygon", "polyline", "line", "ellipse"])
            """for i, path in enumerate(lib):
                if path["id"] == id:
                    return i"""
            return self.id_to_index(xmlid, lib)
        
    def obj_by_id(self, xmlid):
        target_index = self.conv_id(xmlid)
        return self.sobj[target_index]
    

def gen_sigmoid(B_0, B_1, x):
    Z = (B_0+B_1*x)
    return sigmoid(Z)      
def sigmoid(z):
    return 1/(1+math.e**(-z))
def logistic(P):
    return math.log(P/(1-P))
def sigmoid_deriv(z):
    return sigmoid(z)*(1 - sigmoid(z))
mtr_dir = "spd2.svg"
mtr_ref = SVG_Handler(mtr_dir)


class ValueGauge(VMobject):
    def __init__(self, needle,needlehead,semicircle, ticks, curr_angle = 0, **kwargs):
        super().__init__(**kwargs)
        self.needle = needle
        self.curr_angle = curr_angle
        self.needlehead=needlehead
        self.semicircle=semicircle
        self.ticks = ticks
        self.add(self.semicircle, self.needlehead, self.needle)
        for tick in ticks:
            self.add(tick)
    def set_angle(self, theta, about_point):
        delta_theta = theta - self.curr_angle
        self.needle.rotate(angle = delta_theta, about_point=about_point)
        self.curr_angle = theta
        
    def set_angle_clockwise(self, theta, about_point):
        delta_theta = -(theta - self.curr_angle)
        self.needle.rotate(angle = delta_theta, about_point=about_point)
        self.curr_angle = theta
        
class LRpts(Scene):
    def construct(self):
        ndlpin = mtr_ref.obj_by_id("ndlpin")
        ndlhead = mtr_ref.obj_by_id("ndlhead")
        semicircle = mtr_ref.obj_by_id("semicircle")
        tick_ids = mtr_ref.g_id2c("ticks")
        ticks = [mtr_ref.sobj[tick_id].set_color("WHITE") for tick_id in tick_ids]
        ndlpin.rotate(angle = PI/2, about_point=ndlhead.get_center())

        lastxrange=6
        axes_w_points = (
            Axes(
                x_range=[0, lastxrange, 1],
                x_length=9,
                y_range=[0, 1.2, 1],
                y_length=2,
                x_axis_config = {"color":BLUE},
                y_axis_config = {"color":BLUE}
                )
            .add_coordinates())
        x_label = axes_w_points.get_x_axis_label(Tex("Feature Variable").scale(0.8), edge=DOWN, direction=DOWN, buff=0.3)
        y_label = axes_w_points.get_y_axis_label(Tex("Probability").scale(0.7).rotate(90 * DEGREES), edge=LEFT, direction=LEFT, buff=0)
        axes_w_points.add(x_label, y_label)
        dots = []
        coords = [[0.5,0],[1.0,0],[2.1,0],[3.5,0],[3,1],[4,1],[4.6,1],[4.9,1]]
        #coords = [[*coord,0]] for coord in coords]
        dotsizefactor = 1.3
        dotsize = 0.08*dotsizefactor
        for coord in coords:
            point = axes_w_points.c2p(*coord)
            if coord[1] == 0:
                dots.append(Dot(color="GREEN", radius=dotsize).move_to(
                    point
                    ))
            elif coord[1] == 1:
                dots.append(Dot(color="RED",radius=dotsize).move_to(
                    point
                    ))
        coord_group = VGroup(*dots)
        
        fits = []
        guess_fit_coeffs = [[-4,2],[-11,3],[-8,3],[-14,5]]
        for guess_fit_coeff in guess_fit_coeffs:
            B_0, B_1 = guess_fit_coeff
            guess_fit = axes_w_points.plot(
            lambda x: gen_sigmoid(B_0, B_1,x), x_range=[0, lastxrange], color=PINK
            ).set_stroke(width=6)
            guess_fit = DashedVMobject(guess_fit, num_dashes=30)
            fits.append(guess_fit)

        perf_fit1 = axes_w_points.plot(
            lambda x: gen_sigmoid(-6.976, 2.21, x), x_range=[0, lastxrange], color=PINK
            ).set_stroke(width=8)
        fits.append(perf_fit1)

        
        #dot = Dot([lastxrange,1,0])
        line = DashedLine(
            start=axes_w_points.c2p(0, 1),
            end=axes_w_points.c2p(lastxrange, 1)
        ).set_color(YELLOW)
        #===============================================
        # INTRO TITLE ANIMATION
        title_top = Text("THE", font_size=64, weight=BOLD).set_color(PINK)
        title_middle = Text("LOGISTIC REGRESSION", font_size=70, weight=BOLD).set_color_by_gradient(PINK, YELLOW)
        title_bottom = Text("CLASSIFIER", font_size=64, weight=BOLD).set_color(YELLOW)
        title_group = VGroup(title_top, title_middle, title_bottom).arrange(DOWN, buff=0.3)
        
        self.play(FadeIn(title_top, shift=DOWN*0.5), run_time=0.8)
        self.play(FadeIn(title_middle, shift=UP*0.3), run_time=1.2)
        self.play(FadeIn(title_bottom, shift=UP*0.5), run_time=0.8)
        self.wait(0.8)
        self.play(FadeOut(title_group, shift=UP*2, scale=1.5), run_time=1.2)
        self.wait(0.3)
        
        #===============================================
        # EXPLANATION TEXT
        explanation = Text("The Logistic Regression Algorithm predicts the \nprobability of belonging to a class", 
                          font_size=32, color=WHITE).to_edge(UP)
        explanation2 = Text("This algorithm allows us to classify new data by\nfitting a sigmoid curve on existing data\nIt outputs a probability of data\nbeing in class 1", 
                           font_size=32, color=WHITE).to_edge(UP)
        class0_label = Text("CLASS 0", color=GREEN).scale(0.7).move_to((-3,-3,0))
        class1_label = Text("CLASS 1", color=RED).scale(0.7).move_to((3,2,0))
        # Create arrows
        arrow0 = Arrow(
            class0_label.get_top(),
            axes_w_points.c2p(2.5,0),
            buff=0.1,
            color=GREEN
        )

        arrow1 = Arrow(
            class1_label.get_bottom(),
            axes_w_points.c2p(2.5,1),
            buff=0.1,
            color=RED
        )
        sigmoid_label = MathTex(r"\sigma(",r"B_{0}",r";",r"B_{1}",r";x)=\frac{1}{1+e^{-(B_{0}+B_{1}x)}")
        sigmoid_label.set_color(PINK).next_to(axes_w_points, UP).shift(LEFT*2)
        bestscurve = MathTex(r"Best Curve: B_{0}=-6.98,B_{1}=2.21").scale(0.8).move_to(sigmoid_label)
        # Position labels
        
        #============================================== START ANIMATING
        self.play(Create(axes_w_points), run_time=1.3)
        self.play(Write(coord_group), Write(line), run_time=1.3)
        self.play(Write(explanation), run_time=2)
        self.wait(1.5)
        self.play(ReplacementTransform(explanation, explanation2), run_time=1.5)
        self.wait(1)
        self.play(Write(VGroup(arrow0,class0_label)), Write(VGroup(arrow1, class1_label)))
        self.play(FadeOut(explanation2), run_time=1)
        self.play(Write(sigmoid_label))
        #============================================================
        #self.play(Write)
        self.play(Write(fits[0]))
        for i in range(len(fits) - 1):
            if i == len(fits) - 2:
                self.play(ReplacementTransform(
                fits[i], fits[i+1]),ReplacementTransform(sigmoid_label,bestscurve),
                run_time=1.5)
            else:
                self.play(ReplacementTransform(
                    fits[i], fits[i+1]),
                    run_time=1.5)
        self.wait()
        graph = VGroup(axes_w_points, coord_group, line, perf_fit1)
        self.play(graph.animate.to_edge(LEFT), FadeOut(VGroup(bestscurve,arrow0,arrow1,class0_label,class1_label)))
        #=============================================================
        #=============================================================
        #Value tracker
        V = ValueTracker(0)
        #making value gauge instance, adding it and getting the pivot/ turn around point
        ndl = ValueGauge(ndlpin, ndlhead, semicircle, ticks).next_to(graph, RIGHT, buff=0)
        self.play(Create(ndl), run_time=2)
        turn_around_pin = ndl.needlehead.get_center()
        #add updater function 
        ndl.add_updater(
            lambda x: x.set_angle_clockwise(
                PI*gen_sigmoid(-6.976, 2.21, V.get_value()),
                about_point = turn_around_pin
                )
        )
        prob_label = Tex(r"Probability = ", font_size=48*1).set_color(YELLOW)
        prob_value = DecimalNumber(num_decimal_places=3, font_size=48*1).set_color(YELLOW)
        prob_group = VGroup(prob_label, prob_value).arrange(RIGHT, aligned_edge=DOWN).next_to(ndl, DOWN)
        prob_value.add_updater(
            lambda x: x.set_value(gen_sigmoid(-6.976, 2.21, V.get_value()))
        )
        prob_value.shift(UP*0.1)

        
        movingpt1 = always_redraw(
            lambda: Dot(radius=0.08*1.6, color=YELLOW).move_to(
                axes_w_points.c2p(V.get_value(), gen_sigmoid(-6.976, 2.21, V.get_value()))
            )
        )

        #self.add(prob_group)
        self.play(Write(prob_group), Create(movingpt1))
        self.play(V.animate.set_value(lastxrange), run_time = 7)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        
        deriv_text = Text(
            "Some more information:\nThe sigmoid fuction is smooth\nand easily differentiable.",
            font_size=36,
            color=WHITE
        ).to_edge(UP)
        self.play(Write(deriv_text), run_time=2)
        self.play(Indicate(deriv_text))
        self.play(FadeOut(deriv_text, run_time=0.5))
        #======================================================================
        #PART "SHOWING DERIVATIVE"
        #1-all data
        #===================================================================PT2
        axes1 = (
            Axes(
                x_range=[-5,5, 1],
                x_length=9,
                y_range=[0, 1.2, 1],
                y_length=4,
                x_axis_config = {"color":BLUE},
                y_axis_config = {"color":BLUE}
                ).set_stroke(width=5)
            .add_coordinates()
        )
        asymp1 = DashedLine(
            start=axes1.c2p(-5, 1),
            end=axes1.c2p(5, 1),
            dash_length=0.1
            ).set_color(YELLOW)
        sigm = axes1.plot(lambda z: sigmoid(z), x_range=[-5, 5], color=PINK).set_stroke(width=8)
        #sigma_text = MathTex(r"\sigma(Z) = \frac{1}{1+e^{-Z}}").next_to(axes1)
        axes_g1 = VGroup(axes1, sigm, asymp1)
        

        axes2 = (
            Axes(
                x_range=[-5,5, 1],
                x_length=9,
                y_range=[0, 0.5, 1],
                y_length=2,
                x_axis_config = {"color":BLUE},
                y_axis_config = {"color":BLUE}
                ).set_stroke(width=5)
            .add_coordinates()
        )
        sigm_deriv = axes2.plot(lambda P: sigmoid_deriv(P), x_range=[-5, 5], color=PINK).set_stroke(width=8)
        axes_g2 = VGroup(axes2, sigm_deriv)

        sigma_text = MathTex(r"\sigma(Z) = \frac{1}{1+e^{-Z}}").move_to([4,2,0])
        sigma_deriv_text = MathTex(r"\sigma'(Z)=\sigma(Z)(1-\sigma(Z))").next_to(axes2).move_to([4,-2,0])
        #===================================================================================
        #Animation for derivative
        #==================================================================================
        
        self.play(Create(axes1), run_time=1.3)
        self.play(Write(sigm), Write(asymp1), run_time=1.3)
        self.play(axes_g1.animate.to_edge(UP).to_edge(LEFT))
        self.play(Write(sigma_text))
        
        axes_g2.to_edge(DOWN).to_edge(LEFT)
        self.play(Create(axes2))
        self.play(Write(sigm_deriv))
        self.play(Write(sigma_deriv_text))

        k = ValueTracker(-5)
        slope = always_redraw(lambda:axes1.get_secant_slope_group(
            x=k.get_value(),
            graph=sigm,
            dx=0.01,
            secant_line_color=WHITE,
            secant_line_length=5))
        
        movingpt1 = always_redraw(
            lambda: Dot().move_to(axes1.input_to_graph_point(k.get_value(), sigm))
        )
        movingpt2 = always_redraw(
            lambda: Dot().move_to(axes2.input_to_graph_point(k.get_value(), sigm_deriv))
        )
        ln = always_redraw(
            lambda : DashedLine(start = movingpt1.get_bottom(), end = movingpt2.get_top(), dashed_ratio=0.5)
            )
        self.add(slope, movingpt1, movingpt2, ln)
        self.play(
            k.animate.set_value(5),
            rate_func=smooth, run_time=3
            )
        
        #NEW PART STARSt HERE
        #==========================================================================
        #==========================================================================
        to_fade1 = [FadeOut(mob) for mob in self.mobjects]
        self.play(*to_fade1)
#class chkndl(Scene):
    #def construct(self):
        lastxrange=6
        axes_w_points = (
            Axes(
                x_range=[0, lastxrange, 1],
                x_length=9,
                y_range=[0, 1.2, 1],
                y_length=2,
                x_axis_config = {"color":BLUE},
                y_axis_config = {"color":BLUE}
                ))
        x_label = axes_w_points.get_x_axis_label(Tex("Feature Variable").scale(0.8), edge=DOWN, direction=DOWN, buff=0.3)
        y_label = axes_w_points.get_y_axis_label(Tex("Probability").scale(0.7).rotate(90 * DEGREES), edge=LEFT, direction=LEFT, buff=0)
        axes_w_points.add(x_label, y_label)
        dots = []
        coords = [[0.5,0],[1.0,0],[2.1,0],[3.5,0],[3,1],[4,1],[4.6,1],[4.9,1]]
        #coords = [[*coord,0]] for coord in coords]
        dotsizefactor = 1.3
        dotsize = 0.08*dotsizefactor
        for coord in coords:
            point = axes_w_points.c2p(*coord)
            if coord[1] == 0:
                dots.append(Dot(color="GREEN", radius=dotsize).move_to(
                    point
                    ))
            elif coord[1] == 1:
                dots.append(Dot(color="RED",radius=dotsize).move_to(
                    point
                    ))
        coord_group = VGroup(*dots)
        perf_fit1 = axes_w_points.plot(
            lambda x: gen_sigmoid(-6.976, 2.21, x), x_range=[0, lastxrange], color=PINK
            ).set_stroke(width=8)
        axes_w_group = VGroup(axes_w_points, coord_group, perf_fit1)

        sigmoid_label = MathTex(r"\sigma(",r"B_{0}",r";",r"B_{1}",r";x)=\frac{1}{1+e^{-(B_{0}+B_{1}x)}").scale(0.8)
        sigmoid_label.set_color(PINK).next_to(axes_w_points, UP).shift(LEFT*2)

        y_label2 = axes_w_points.get_y_axis_label(Tex("P").scale(0.7).rotate(90 * DEGREES), edge=LEFT, direction=LEFT, buff=0)
    
        desctext1 = Text(
            "Why use sigmoid curve specifically?\nIt is the inverse of the logit function\nIt is the function that linearizes the log(odds) curve",
            font_size=36,
            color=WHITE
        ).to_edge(UP)
        desctext2 = Tex(
            r"Change domain of this curve to log(odds):"
        ).to_edge(UP)
        
        desctext3 = MathTex(
            r"P\rightarrow Log(\frac{P}{1-P})"
        )
        self.play(Create(axes_w_points),Write(coord_group), run_time=1.3
            )
        self.play(Write(perf_fit1), run_time=1.3)
        self.play(Write(desctext1),run_time=2)
        self.play(Indicate(desctext1, color=RED, scale_factor=1.3), run_time=1.5)
        self.play(ReplacementTransform(desctext1,desctext2), run_time=2)
        #self.play(Write(sigmoid_label))
        self.play(Transform(y_label, y_label2))
        self.play(axes_w_group.animate.scale(1).to_edge(UP), FadeOut(desctext2))
        #self.play(Write(desctext3))
        self.remove(x_label)

        #========================================================
        axes3 = (
            Axes(
                x_range=[0, lastxrange, 1],
                x_length=9,
                y_range=[-7, 7, 1],
                y_length=4,
                x_axis_config = {"color":BLUE},
                y_axis_config = {"color":BLUE, "include_numbers":False, "include_tip":False}
                )).to_edge(DOWN)
        y_label_ax3 = axes3.get_y_axis_label(MathTex(r"Log( \frac{P}{1-P})").scale(0.7).rotate(90 * DEGREES), edge=LEFT, direction=LEFT, buff=0)
        axes3.add(y_label_ax3)
        pos_inf_ln = DashedLine(
            start=axes3.c2p(0,8),
            end=axes3.c2p(6, 8),
            dash_length=0.1
            ).set_color(YELLOW)
        pos_inf_label = MathTex(r"+\infty").next_to(pos_inf_ln, LEFT).set_color(YELLOW)
    
        neg_inf_ln = DashedLine(
            start=axes3.c2p(0,-8),
            end=axes3.c2p(6, -8),
            dash_length=0.1
            ).set_color(YELLOW)
        neg_inf_label = MathTex(r"-\infty").next_to(neg_inf_ln, LEFT).set_color(YELLOW)
        #____________________________________________________________________all plotting stuffs
        #===================================================================
        logitplot1 = axes3.plot(
            lambda x: -6.976 + 2.21*x, x_range=[0, lastxrange], color=PINK
            ).set_stroke(width=8)
        #coords = [[0.5,0],[1.0,0],[2.1,0],[3.5,0],[3,1],[4,1],[4.6,1],[4.9,1]]
        dots2 = []
        dotsize = 0.08*1.3
        for coord in coords:
            if coord[1] == 0:
                point = axes3.c2p(coord[0], -8)
                dots2.append(Dot(color="GREEN", radius=dotsize).move_to(
                    point
                    ))
            elif coord[1] == 1:
                point = axes3.c2p(coord[0], +8)
                dots2.append(Dot(color="RED",radius=dotsize).move_to(
                    point
                    ))
        coord_group2 = VGroup(*dots2)
        #======proceed anims
        self.play(Create(axes3), run_time=1.3)
        self.play(Write(logitplot1), run_time = 1.3)
        self.play(Write(pos_inf_ln), Write(neg_inf_ln), Write(pos_inf_label), Write(neg_inf_label), run_time=2.5)
        to_transform = []
        for i in range(len(coords)):
            to_transform.append(ReplacementTransform(coord_group[i].copy(), coord_group2[i]))
        self.play(*to_transform, run_time=1.5)
        self.play(
            #Indicate(VGroup(pos_inf_ln, pos_inf_label), color=RED, scale_factor=1.3), 
            #Indicate(VGroup(neg_inf_ln, neg_inf_label), color=GREEN, scale_factor=1.3),
            Indicate(pos_inf_label, color=RED, scale_factor=1.5),
            Indicate(neg_inf_label, color=GREEN, scale_factor=1.5),
            run_time=2)
        ctr = axes_w_group.get_center()
        all_mobs = VGroup(*self.mobjects)
        self.play(all_mobs.animate.to_edge(RIGHT))

        left_col_width = 3.9
        question_text = Text(
            "Why are classes 0 and 1\nat -inf and inf\nrespectively?",
            font_size=30,
            color=WHITE,
            line_spacing=1.0
        )
        question_text.scale_to_fit_width(left_col_width)
        question_text.to_edge(LEFT, buff=0.22).to_edge(UP, buff=0.45)

        limit_tex_1 = MathTex(
            r"\begin{aligned}"
            r"\lim_{P\to 1}\log\left(\frac{P}{1-P}\right)\\"
            r"=\infty"
            r"\end{aligned}"
        ).scale(0.67)
        limit_tex_2 = MathTex(
            r"\begin{aligned}"
            r"\lim_{P\to 0}\log\left(\frac{P}{1-P}\right)\\"
            r"=-\infty"
            r"\end{aligned}"
        ).scale(0.67)
        limit_group = VGroup(limit_tex_1, limit_tex_2).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        limit_group.scale_to_fit_width(left_col_width)
        limit_group.to_edge(LEFT, buff=0.22).to_edge(UP, buff=0.45)

        conclusion_text = Text(
            "Therefore this forms a\nstraight line extending\ndomain to -inf and inf.",
            font_size=30,
            color=WHITE,
            line_spacing=1.0
        )
        conclusion_text.scale_to_fit_width(left_col_width)
        conclusion_text.to_edge(LEFT, buff=0.22).to_edge(UP, buff=0.45)

        coeff_text_lines = VGroup(
            MathTex(r"\text{The coefficients } B_0, B_1 \text{ characterise}"),
            MathTex(r"\text{the straight line } Z = B_0 + B_1 x."),
            MathTex(r"\text{(i.e. they correspond to the intercept}"),
            MathTex(r"\text{and slope of the line).}")
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        coeff_text_lines.scale(0.75)


        self.play(Write(question_text), run_time=2)
        self.wait()
        self.play(ReplacementTransform(question_text, limit_group), run_time=2)
        self.wait(2.2)
        self.play(ReplacementTransform(limit_group, conclusion_text), run_time=2)
        self.wait()
        self.play(FadeOut(VGroup(
            conclusion_text, axes3, pos_inf_ln, neg_inf_ln, pos_inf_label, neg_inf_label, logitplot1, coord_group2
        )), run_time=1.5)
        

        self.play(axes_w_group.animate.move_to(ctr), run_time=1.8)
        coeff_text_lines.move_to(ORIGIN)
        coeff_text_lines.to_edge(DOWN, buff=0.5)
        self.play(Write(coeff_text_lines), run_time=2)
        self.wait(1)

        coeff_calc_text = Tex(
            r"\text{How do we find these coefficients?}\\"
            r"\text{By maximizing the likelihood function.}",
            tex_environment="flushleft"
        ).scale(0.8)
        coeff_calc_text.move_to(coeff_text_lines)
        
        likelihood_steps_text = Tex(
            r"\text{Take values of } $B_{0}$ \text{ and } $B_{1}$.\\"
            r"\text{Compute the probability each point is classified correctly.}\\"
            r"\text{Multiply all those probabilities (likelihood).}",
            tex_environment="flushleft"
        ).scale(0.75)
        likelihood_steps_text.move_to(coeff_text_lines)


        self.play(ReplacementTransform(coeff_text_lines, coeff_calc_text), run_time=2)
        self.wait()
        self.play(ReplacementTransform(coeff_calc_text, likelihood_steps_text), run_time=2)
        self.wait(1)

        dashedlines =[]
        for i in range(len(coords)):
            dot_i = coord_group[i]
            coord = coords[i]
            startpt = dot_i.get_center()
            endpt = axes_w_points.c2p(coord[0], gen_sigmoid(-6.976, 2.21, coord[0]))
            dashline = DashedLine(
                start=startpt,
                end=endpt,
                dash_length=0.1).set_color(YELLOW)
            dashedlines.append(dashline)
        dashedlines_group = VGroup(*dashedlines)
        self.play(Write(dashedlines_group), run_time=3)

        moving_pts_animation = []
        moving_pt_dt = 1.5
        for i in range(len(coords)):
            #dot_i = coord_group[i]
            coord = coords[i]
            endpt = axes_w_points.c2p(coord[0], gen_sigmoid(-6.976, 2.21, coord[0]))
            moving_pts_animation.append(coord_group[i].animate.move_to(endpt))
        self.play(
            FadeOut(dashedlines_group),
            *moving_pts_animation, run_time = moving_pt_dt
        )


        new_sigmoid_label = MathTex(r"\sigma(",r"B_{0}",r";",r"B_{1}",r";x)=\frac{1}{1+e^{-(B_{0}+B_{1}x)}").scale(0.8).set_color(YELLOW).next_to(axes_w_points, DOWN)
        self.play(Write(new_sigmoid_label), run_time = 2)
        #==============================================================
        #coords = [[0.5,0],[1.0,0],[2.1,0],[3.5,0],[3,1],[4,1],[4.6,1],[4.9,1]]
        to_multiply = []
        coord_map = [[coords[i], i] for i in range(len(coords))]
        coords_map_sorted = sorted(coord_map,key=lambda x: x[0][0])
        #sorted_dots = [coord_map_elem[1] for coord_map_elem in coords_map_sorted]

        for k, coord_map_i in enumerate(coords_map_sorted):
            
            #dot_i = coord_map_i[1]
            dot_xval = coord_map_i[0][0]
            dot_class = coord_map_i[0][1]
            if k == len(coords_map_sorted) - 1:
                multiply_operator = ""
            else:
                multiply_operator = r"\cdot"
            if dot_class == 1:
                append_text = fr"\sigma({dot_xval}) {multiply_operator} "
            elif dot_class == 0:
                append_text = fr"(1-\sigma({dot_xval})) {multiply_operator} "

            to_multiply.append(append_text)
                

        to_multiply_mathtex = MathTex(*to_multiply).scale(0.7).next_to(new_sigmoid_label, DOWN).set_color(PINK)
        indicate_dt = 0.5
        write_dt = 0.5
        for i, coord_map_i in enumerate(coords_map_sorted):
            dot_idx = coord_map_i[1]
            self.play(
                Indicate(coord_group[dot_idx], color=YELLOW, scale_factor=2.8, run_time=indicate_dt)
            )
            self.play(Write(to_multiply_mathtex[i]), run_time=write_dt)

        
        #=====================================================================================================
        #=====================================================================================================
        self.play(likelihood_steps_text.animate.to_edge(DOWN))
        last_text1 = Tex(
            r"\text{Take new values of } $B_{0}$ \text{ and } $B_{1}$, \text{Repeat to maximise Log(Likelihood)}\\"
            r"\text{It is easier to maximise } $\log\mathcal{L}(\theta)$ \text{ than } $\mathcal{L}(\theta)$\\"
            r"\text{We use Gradient Descent, Newton's Method, or BFGS, etc.}",
            tex_environment="flushleft"
        ).scale(0.75)
        last_text1.move_to(likelihood_steps_text)
        max_likelihood_tex = (
            MathTex(r"\mathcal{L}(B_{0},B_{1}) = \prod_{i=1}^{n} \sigma(x_i)^{t_i}(1-\sigma(x_i))^{1-t_i}")
            .scale(0.72)
            .set_color(YELLOW)
            .move_to(to_multiply_mathtex)
        ).shift(DOWN*0.3)
        t_explanation = Tex(
            r"t simply refers to the\\"
            r"class label.(1 or 0)").next_to(max_likelihood_tex, RIGHT).scale(0.6).to_edge(RIGHT).set_color(PINK)
        
        #=====================================================================================================


        self.play(ReplacementTransform(likelihood_steps_text,last_text1), run_time=2)
        self.play(ReplacementTransform(to_multiply_mathtex, max_likelihood_tex), run_time=2.5)
        self.play(Write(t_explanation))
        self.wait()
        log_likelihood_tex = (
            MathTex(r"\log\mathcal{L}(B_{0},B_{1}) = \sum_{i=1}^{n} \big[t_i\log\sigma(x_i) + (1-t_i)\log(1-\sigma(x_i))\big]")
            .scale(0.72)
            .set_color(YELLOW)
            .move_to(max_likelihood_tex)
        )
        self.play(
            ReplacementTransform(max_likelihood_tex, log_likelihood_tex),
            FadeOut(t_explanation),
            run_time=2.5,
        )

        log_likelihood_box = SurroundingRectangle(log_likelihood_tex, buff=0.3, color=PINK)
        log_likelihood_box.add_updater(
            lambda mob: mob.become(SurroundingRectangle(log_likelihood_tex, buff=0.3, color=PINK))
        )
        self.play(Create(log_likelihood_box))
        #ENDS HERE------------------------!!!!!!!!!!!
        #ALLSUBSCRIBE TEXT ANDSTUFF
        MYLOGO = SVGMobject("Leaflogoss.svg").scale_to_fit_height(config.frame_height/1.5)
        to_fade = []
        MYLOGO.fadekey="1"#content doesnt matter
        for m in self.mobjects:
            if not hasattr(m, "fadekey"):
                to_fade.append(FadeOut(m))
        
        self.play(*to_fade,
                  FadeIn(MYLOGO),
                  run_time=2
                  )

        channelname = Tex("MathMinter").set_color("#f542cb").scale(3).set_stroke(width=5)#other color #111e3b #b2ca2c
        channelname.next_to(MYLOGO, UP)

        subscribetext = Tex("SUBSCRIBE").set_color("#f542cb").scale(2).set_stroke(width=4).next_to(MYLOGO, DOWN)
        subscribebox = Rectangle(width=subscribetext.width +0.2, height=subscribetext.height+0.2).set_stroke(color="#b2ca2c", width=12)
        subscribebox.move_to(subscribetext)

        self.play(Write(channelname), Write(subscribetext), Create(subscribebox))
        self.play(VGroup(subscribetext, subscribebox).animate.scale(2), run_time=0.5)
        self.play(VGroup(subscribetext, subscribebox).animate.scale(1/2), run_time=0.5)
        self.wait(1.5)
        self.wait()

        