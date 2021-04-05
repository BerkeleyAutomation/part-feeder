const planck = require('planck-js');
planck.Settings.maxPolygonVertices = 50;
import * as math from "mathjs";

function generateRotationMatrix(angle) {
    let s = math.sin(angle);
    let c = math.cos(angle);

    return [[c, -s], [s, c]];
}

function range(size, startAt = 0) {
    return [...Array(size).keys()].map(i => i + startAt);
}

function zip(a, b) {
    return a.map((k, i) => [k, b[i]]);
}

function mod2PI(theta) {
    return math.mod(theta, 2 * Math.PI);
}

function generateBoundedCallable(boundedPiecewiseFunc, period) {
    let _min = boundedPiecewiseFunc[0][0];
    let _max = boundedPiecewiseFunc[boundedPiecewiseFunc.length - 1][1];

    let f = function (theta) {
        while (theta > _max || math.equal(theta, _max)) {
            theta -= period;
        }
        while (theta < _min && !math.equal(theta, _min)) {
            theta += period;
        }
        for (const [lower, upper, l, i] of boundedPiecewiseFunc) {
            if (math.equal(lower, theta) || (lower < theta && theta < upper)) {
                return l * Math.abs(Math.cos(theta - i));
            }
        }
    }

    return f;
}

function generateTransferExtremaCallable(transferFunc, period) {
    let _min = transferFunc[0][0], _max = transferFunc[transferFunc.length - 1][1];

    return function (theta) {
        while (theta > _max || math.equal(theta, _max)) {
            theta -= period;
        }
        while (theta < _min && !math.equal(theta, _min)) {
            theta += period;
        }

        for (const [a, b, t] of transferFunc) {
            if ((a < theta && theta < b) || math.equal(theta, a)) {
                return t;
            }
        }
    }
}

class Gripper {

    constructor(world, x, y, angle, distance = 200, length = 200, velocity = 40) {
        this.angle = angle
        length /= 2;
        distance /= 2;

        // gripper position vectors, vector in column 0 is the bottom gripper
        let rot_mat = generateRotationMatrix(angle);
        let offset_mat = math.multiply(rot_mat, math.matrix([[0, 0], [-distance, distance]]));
        let pos_mat = math.add(offset_mat, [[x, x], [y, y]]);

        // velocity of the bottom gripper (top gripper is just -vel_vec)
        let vel_vec = math.subtract(pos_mat.subset(math.index([0, 1], 1)), pos_mat.subset(math.index([0, 1], 0)));
        vel_vec = new planck.Vec2(math.subset(vel_vec, math.index(0, 0)), math.subset(vel_vec, math.index(1, 0)))
        vel_vec.normalize();
        vel_vec.mul(velocity);


        // bottom gripper
        this.botVel = vel_vec
        this.botPos = new planck.Vec2(math.subset(pos_mat, math.index(0, 0)), math.subset(pos_mat, math.index(1, 0)));
        [this.bot, this.botEdge] = this.makeGripper(world, this.botPos, length, angle);

        this.topVel = planck.Vec2.neg(this.botVel);
        this.topPos = new planck.Vec2(math.subset(pos_mat, math.index(0, 1)), math.subset(pos_mat, math.index(1, 1)));
        [this.top, this.topEdge] = this.makeGripper(world, this.topPos, length, angle);
    }

    makeGripper(world, pos, length, angle, height = 5) {
        let body = world.createBody({
            type: 'kinematic',
            position: pos,
            angle: angle,
            fixedRotation: true,
        })

        let edge = planck.Edge(planck.Vec2(-length, 0), planck.Vec2(length, 0));

        body.createFixture({
            shape: edge,
            friction: 0,
            restitution: 0,
            density: 1,
            // Gripper collision filter
            filterGroupIndex: -1,
            filterCategoryBits: 0b01,
            filterMaskBits: 0b10
        });

        return [body, edge];
    }

    squeeze() {
        this.bot.setLinearVelocity(this.botVel);
        this.top.setLinearVelocity(this.topVel);
    }

    unsqueeze() {
        this.top.setLinearVelocity(planck.Vec2.neg(this.topVel));
        this.bot.setLinearVelocity(planck.Vec2.neg(this.botVel));
    }

    push() {
        this.bot.setLinearVelocity(this.botVel);
    }

    grasp() {
        this.top.setLinearVelocity(this.topVel);
    }

    stop() {
        this.top.setLinearVelocity(new planck.Vec2(0, 0));
        this.bot.setLinearVelocity(new planck.Vec2(0, 0));
    }

    limitUnsqueeze() {
        let eps = 10;

        if (planck.Vec2.lengthOf(planck.Vec2.sub(this.top.getPosition(), this.topPos)) < eps) {
            this.top.setLinearVelocity(new planck.Vec2(0, 0));
        }

        if (planck.Vec2.lengthOf(planck.Vec2.sub(this.bot.getPosition(), this.botPos)) < eps) {
            this.bot.setLinearVelocity(new planck.Vec2(0, 0));
        }
    }

    distance() {
        return planck.Vec2.lengthOf(planck.Vec2.sub(this.top.getPosition(), this.bot.getPosition()));
    }
}

class Polygon {

    constructor(world, x, y, points, angle = null) {
        // points needs to be an array of {x: number, y: number} objects

        if (angle == null) {
            angle = Math.random() * Math.PI;
        }

        this.body = world.createBody({
            type: "dynamic",
            position: planck.Vec2(x, y),
            angle: angle,
        });

        this.poly = planck.Polygon(points);

        this.body.createFixture({
            shape: this.poly,
            density: 1,
            friction: 0,
            restitution: 0
        });
    }

    setMoveFilter() {
        for (let f = this.body.getFixtureList(); f; f = f.getNext()) {
            if (!f.isSensor()) {
                f.setFilterData({
                    groupIndex: -2,
                    categoryBits: 0b10,
                    maskBits: 0b00
                });
            }
        }
    }

    setSqueezeFilter() {
        for (let f = this.body.getFixtureList(); f; f = f.getNext()) {
            if (!f.isSensor()) {
                f.setFilterData({
                    groupIndex: -2,
                    categoryBits: 0b10,
                    maskBits: 0b11
                });
            }
        }
    }

    move() {
        this.body.setLinearVelocity(planck.Vec2(100, 0));
        this.setMoveFilter();
    }

    squeeze() {
        this.setSqueezeFilter();
    }
}

class Display {
    constructor(points, gripperAngles, polygonAngles) {
        this.rows = [-150, 150];
        this.points = points;
        this.angles = gripperAngles;
        this.polygonAngles = polygonAngles;

        this.spacing = 300;
        this.gripperPos = range(gripperAngles.length, 1).map(i => i * this.spacing);
        this.startPos = 0;
        this.displayPos = (gripperAngles.length + 1) * this.spacing;
        this.delPos = (gripperAngles.length + 2) * this.spacing;

        this.xlim = [this.startPos, this.delPos];
        this.ylim = [-300, 400];

        this.world = planck.World({
            gravity: new planck.Vec2(0, 0)
        });

        this.grippers = range(this.rows.length).map(i => Array(gripperAngles.length));
        this.initGrippers();

        this.polygons = range(this.rows.length).map(i => Array(0));

        this.initDrawPoints();
        this.setPolygons();

        this.TOTAL_TIME = null;
    }

    initGrippers() {
        for (const [row, ypos] of this.rows.entries()) {
            for (const [i, [angle, xpos]] of zip(this.angles, this.gripperPos).entries()) {
                this.grippers[row][i] = new Gripper(this.world, xpos, ypos, angle);
            }
        }
    }

    initDrawPoints() {
        let drawPoints = {x: this.points.map(v => v.x), y: this.points.map(v => v.y)};
        drawPoints.x.push(drawPoints.x[0]);
        drawPoints.y.push(drawPoints.y[0]);

        let line = {x: drawPoints.x.slice(2), y: drawPoints.y.slice(2)};
        this.thickLine = 0;
        let dist = function (points) {
            return planck.Vec2.lengthOf(planck.Vec2.sub(
                planck.Vec2(points.x[0], points.y[0]),
                planck.Vec2(points.x[1], points.y[1])
            ));
        }
        for (let i = 0; i < drawPoints.length - 1; i++) {
            let candidate = {x: drawPoints.x.slice(i, i + 2), y: drawPoints.y.slice(i, i + 2)};
            if (dist(candidate) > dist(line)) {
                line = candidate;
                this.thickLine = i;
            }
        }

        this.drawPoints = [drawPoints.x, drawPoints.y];
    }

    setPolygons() {
        if (this.rows.length === this.polygonAngles.length) {
            for (let i = 0; i < this.rows.length; i++) {
                this.polygons[i] = [];
                for (const [index, angle] of this.polygonAngles[i].entries()) {
                    this.polygons[i].push(new Polygon(this.world, (index + 1) * this.spacing, this.rows[i],
                        this.points, parseFloat(angle)));
                }
            }
        }
    }

    addPolygon() {
        for (const [row, ypos] of this.rows.entries()) {
            this.polygons[row].unshift(new Polygon(this.world, 0, ypos, this.points));
        }
    }

    stepEngine() {
        let timeStep = 1 / 60;
        let velocityIterations = 10;
        let positionIterations = 8;
        this.world.step(timeStep, velocityIterations, positionIterations);
    }

    step(dt) {
        throw "Redefine in subclass!";
    }

    draw() {
        let traces = this.getTraces();

        let fig = {
            data: traces,
            layout: {
                xaxis: {
                    range: this.xlim,
                    showticklabels: false,
                    showgrid: false,
                    zeroline: false
                },
                yaxis: {
                    scaleanchor: "x",
                    range: this.ylim,
                    showticklabels: false,
                    showgrid: false,
                    zeroline: false
                },
                showlegend: false,
                annotations: zip(this.gripperPos, this.angles).map(([g, a]) => ({
                    x: g,
                    y: 350,
                    text: Math.round(a * 180 / Math.PI),
                    showarrow: false,
                    font: {
                        family: "sans serif",
                        size: 18,
                        color: "black"
                    }
                }))
            }
        };

        return fig;
    }

    getTraces() {
        let traces = [];

        for (let i = 0; i < this.rows.length; i++) {
            for (let j = 0; j < this.polygons[i].length; j++) {
                let p = this.polygons[i][j].body;
                let drawPoints = math.multiply(generateRotationMatrix(p.getAngle()), this.drawPoints);
                drawPoints = math.add(drawPoints, [Array(this.drawPoints[0].length).fill(p.getPosition().x),
                    Array(this.drawPoints[1].length).fill(p.getPosition().y)]);
                // Polygon
                let poly = {
                    type: "scatter",
                    x: drawPoints[0],
                    y: drawPoints[1],
                    mode: "lines",
                    fill: "toself",
                    fillcolor: "#2D4262",
                    line: {
                        color: "black"
                    }
                }

                // Thick alignment line
                let line = {
                    type: "scatter",
                    x: math.squeeze(math.subset(drawPoints, math.index(0, math.range(this.thickLine, this.thickLine + 2)))),
                    y: math.squeeze(math.subset(drawPoints, math.index(1, math.range(this.thickLine, this.thickLine + 2)))),
                    mode: "lines",
                    line: {
                        color: "#DB9501",
                        width: 4
                    }
                }

                traces.push(poly, line);
            }

            for (let j = 0; j < this.grippers[i].length; j++) {
                let g = this.grippers[i][j];
                let drawPointsTop = math.transpose([g.topEdge.m_vertex1, g.topEdge.m_vertex2].map(v => [v.x, v.y]));
                drawPointsTop = math.add(
                    math.multiply(
                        generateRotationMatrix(g.top.getAngle()),
                        drawPointsTop
                    ),
                    [
                        Array(drawPointsTop[0].length).fill(g.top.getPosition().x),
                        Array(drawPointsTop[1].length).fill(g.top.getPosition().y)
                    ]
                );

                let drawPointsBot = math.transpose([g.botEdge.m_vertex1, g.botEdge.m_vertex2].map(v => [v.x, v.y]));
                drawPointsBot = math.add(
                    math.multiply(generateRotationMatrix(g.bot.getAngle()), drawPointsBot),
                    [
                        Array(drawPointsBot[0].length).fill(g.bot.getPosition().x),
                        Array(drawPointsBot[1].length).fill(g.bot.getPosition().y)
                    ]
                );

                let top = {
                    type: "scatter",
                    x: drawPointsTop[0],
                    y: drawPointsTop[1],
                    mode: "lines",
                    line: {
                        color: "black",
                        width: 4
                    }
                }

                let bot = {
                    type: "scatter",
                    x: drawPointsBot[0],
                    y: drawPointsBot[1],
                    mode: "lines",
                    line: {
                        color: "black",
                        width: 4
                    }
                }

                traces.push(top, bot);
            }
        }

        return traces;
    }

    stepDraw(loops = 1) {
        let figs = [];
        for (let i = 0; i < loops; i++) {
            for (let j = 0; j < this.TOTAL_TIME; j++) {
                this.step(j);
                if (j % 16 === 0) {
                    figs.push(this.draw());
                }
            }
        }

        return figs;
    }

    serialize() {
        return this.polygons.map(row => row.map(p => p.body.getAngle()));
    }
}

class SqueezeDisplay extends Display {

    constructor(points, gripperAngles, polygonAngles, diameterFunc, squeezeFunc) {
        super(points, gripperAngles, polygonAngles);

        this.gripperMinDist = range(this.rows.length).map(i => Array(0));
        this.polygonRotateDist = range(this.rows.length).map(i => Array(0));
        this.stopRotateAngle = range(this.rows.length).map(i => Array(0));

        this.diameterCallable = generateBoundedCallable(diameterFunc, Math.PI);
        this.squeezeCallable = generateTransferExtremaCallable(squeezeFunc, Math.PI);

        this.TOTAL_TIME = 450;
    }

    step(dt) {
        dt = dt % this.TOTAL_TIME;

        if (dt === 0) {
            this.addPolygon();

            for (const row of this.polygons) {
                for (const p of row) {
                    p.body.setLinearDamping(0);
                    p.move();
                }
            }

            for (const row of this.grippers) {
                for (const g of row) {
                    g.stop();
                }
            }
        } else if (0 + 50 < dt && dt < 200) {
            // move phase
            for (const row of this.polygons) {
                if (row.length) {
                    let last = row[row.length - 1].body;
                    if (last.getPosition().x >= this.delPos) {
                        this.world.destroyBody(last);
                        row.pop();
                    }

                    let eps = 3;
                    for (const p of row) {
                        if (this.gripperPos.some(g => Math.abs(p.body.getPosition().x - g) < eps) ||
                            Math.abs(p.body.getPosition().x - this.displayPos) < eps) {
                            p.body.setLinearVelocity(planck.Vec2(0, 0));
                        }
                    }
                }

            }
        } else if (dt === 200) {
            // init squeeze phase
            this.gripperMinDist = this.rows.map(_ => this.gripperPos.map(_ => 50));
            this.polygonRotateDist = this.rows.map(_ => this.gripperPos.map(_ => 0));
            this.stopRotateAngle = this.rows.map(_ => this.gripperPos.map(_ => 0));

            for (const [rowIdx, row] of this.polygons.entries()) {
                for (const [i, p] of row.slice(0, this.angles.length).entries()) {
                    p.squeeze();
                    p.body.setFixedRotation(true);

                    let gripperAngle = this.grippers[rowIdx][i].angle;

                    // Angle of gripper relative to polygon
                    let relAngle = mod2PI(mod2PI(gripperAngle) - mod2PI(p.body.getAngle()));
                    this.polygonRotateDist[rowIdx][i] = this.diameterCallable(relAngle);

                    let relOutputAngle = this.squeezeCallable(relAngle);
                    let worldOutputAngle = mod2PI(mod2PI(gripperAngle) - mod2PI(relOutputAngle));
                    this.stopRotateAngle[rowIdx][i] = worldOutputAngle;

                    this.gripperMinDist[rowIdx][i] = this.diameterCallable(relOutputAngle);
                }
            }

            for (const row of this.grippers) {
                for (const g of row) {
                    g.squeeze();
                }
            }
        } else if (200 < dt && dt < 350) {
            for (const [rowIdx, row] of this.grippers.entries()) {
                for (const [i, g] of row.entries()) {
                    let distance = g.distance();

                    if (i < this.polygons[rowIdx].length) {
                        let p = this.polygons[rowIdx][i];
                        if (Math.abs(distance - this.polygonRotateDist[rowIdx][i]) < 10) {
                            p.body.setFixedRotation(false);
                            p.body.setAngularDamping(0);
                        }

                        if (Math.abs(mod2PI(p.body.getAngle()) - mod2PI(this.stopRotateAngle[rowIdx][i])) < 0.08) {
                            p.body.setFixedRotation(true);
                            p.body.setLinearDamping(Infinity);
                            p.setMoveFilter();

                            let collideTop = false;
                            this.world.rayCast(g.top.getWorldPoint(g.topEdge.m_vertex1),
                                g.top.getWorldPoint(g.topEdge.m_vertex2),
                                function (fixture, point, normal, fraction) {
                                    if (fixture === p.body.getFixtureList()) {
                                        collideTop = true;
                                        return 0;
                                    }
                                    return -1;
                                });

                            let collideBot = false;
                            this.world.rayCast(g.bot.getWorldPoint(g.botEdge.m_vertex1),
                                g.bot.getWorldPoint(g.botEdge.m_vertex2),
                                function (fixture, point, normal, fraction) {
                                    if (fixture === p.body.getFixtureList()) {
                                        collideBot = true;
                                        return 0;
                                    }
                                    return -1;
                                });

                            if (collideTop) {
                                g.top.setLinearVelocity(planck.Vec2(0, 0));
                            }
                            if (collideBot) {
                                g.bot.setLinearVelocity(planck.Vec2(0, 0));
                            }
                            if (collideTop && collideBot) {
                                p.body.setAngle(this.stopRotateAngle[rowIdx][i]);
                            }
                        }
                    }
                    if (Math.abs(distance - this.gripperMinDist[rowIdx][i]) < 5 || distance < 3) {
                        g.stop();
                        if (i < this.polygons[rowIdx].length) {
                            let p = this.polygons[rowIdx][i]
                            p.body.setAngle(this.stopRotateAngle[rowIdx][i]);
                            p.body.setFixedRotation(true);
                            p.body.setLinearDamping(Infinity);
                        }
                    }
                }
            }
        } else if (dt === 350) {
            for (const row of this.polygons) {
                for (const p of row) {
                    p.body.setFixedRotation(true);
                    p.body.setLinearDamping(Infinity);
                }
            }
            for (const row of this.grippers) {
                for (const g of row) {
                    g.unsqueeze();
                }
            }
        } else if (dt > 350) {
            for (const row of this.grippers) {
                for (const g of row) {
                    g.limitUnsqueeze();
                }
            }
        }

        this.stepEngine();
    }
}

class PushGraspDisplay extends Display {
    constructor(points, gripperAngles, polygonAngles, radiusFunc, diameterFunc, pushFunc, pushGraspFunc) {
        super(points, gripperAngles, polygonAngles);

        this.gripperPushDist = this.rows.map(r => Array(0));
        this.gripperSqueezeDist = this.rows.map(r => Array(0));

        this.stopPushAngle = this.rows.map(r => Array(0));
        this.stopSqueezeAngle = this.rows.map(r => Array(0));


        this.polygonPins = this.rows.map(r => Array(0));

        this.radiusCallable = generateBoundedCallable(radiusFunc);
        this.diameterCallable = generateBoundedCallable(diameterFunc);
        this.pushCallable = generateTransferExtremaCallable(pushFunc);
        this.pushGraspCallable = generateTransferExtremaCallable(pushGraspFunc);

        this.staticBody = this.world.createBody({type: 'static'});
        this.TOTAL_TIME = 650;
    }

    step(dt) {
        dt = dt % this.TOTAL_TIME;

        if (dt === 0) {
            this.addPolygon();

            for (const row of this.polygons) {
                for (const p of row) {
                    p.body.setLinearDamping(0);
                    p.move();
                }
            }

            for (const row of this.grippers) {
                for (const g of row) {
                    g.stop();
                }
            }
        } else if (50 < dt && dt < 200) {
            for (const row of this.polygons) {
                if (row.length) {
                    let last = row[row.length - 1].body;
                    if (last.getPosition().x >= this.delPos) {
                        this.world.destroyBody(last);
                        row.pop();
                    }

                    let eps = 3;
                    for (const p of row) {
                        if (this.gripperPos.some(g => Math.abs(p.body.getPosition().x - g) < eps) ||
                            Math.abs(p.body.getPosition().x - this.displayPos) < eps) {
                            p.body.setLinearVelocity(planck.Vec2(0, 0));
                        }
                    }
                }

            }
        } else if (dt === 200) {
            this.gripperPushDist = this.rows.map(r => Array(this.gripperPos.length).fill(50));
            this.gripperSqueezeDist = this.rows.map(r => Array(this.gripperPos.length).fill(50));

            this.stopPushAngle = this.rows.map(r => Array(this.polygons[0].length).fill(0));
            this.stopSqueezeAngle = this.rows.map(r => Array(this.polygons[0].length).fill(0));

            this.polygonPins = this.rows.map(r => Array(Math.min(this.polygons[0].length, this.gripperPos.length)));

            for (const [rowIdx, row] of this.polygons.entries()) {
                for (const [i, p] of row.slice(0, this.gripperPos.length).entries()) {
                    p.squeeze();
                    p.body.setFixedRotation(false);

                    let g = this.grippers[rowIdx][i];

                    let relAngle = mod2PI(mod2PI(g.angle) - mod2PI(p.body.getAngle()));
                    let relPushOutputAngle = this.pushCallable(relAngle);
                    let worldPushOutputAngle = mod2PI(mod2PI(g.angle) - mod2PI(relPushOutputAngle));

                    let relPGOutputAngle = this.pushGraspCallable(relAngle);
                    let worldPGOutputAngle = mod2PI(mod2PI(g.angle) - mod2PI(relPGOutputAngle));

                    this.gripperPushDist[rowIdx][i] = this.radiusCallable(relPushOutputAngle);
                    this.gripperSqueezeDist[rowIdx][i] = this.diameterCallable(relPGOutputAngle);

                    this.stopPushAngle[rowIdx][i] = worldPushOutputAngle;
                    this.stopSqueezeAngle[rowIdx][i] = worldPGOutputAngle;

                    let pin = this.world.createJoint(
                        planck.RevoluteJoint({}, this.staticBody, p.body, p.body.getWorldCenter()));
                    this.polygonPins[rowIdx][i] = pin;
                }
            }

            for (const row of this.grippers) {
                for (const g of row) {
                    g.push();
                }
            }
        } else if (200 < dt && dt < 350) {
            // push phase
            for (const [rowIdx, row] of this.grippers.entries()) {
                for (const [i, g] of row.entries()) {
                    if (i < this.polygons[rowIdx].length) {
                        let p = this.polygons[rowIdx][i];
                        let distance = planck.Vec2.lengthOf(planck.Vec2.sub(g.bot.getPosition(), p.body.getWorldCenter()));
                        if (Math.abs(mod2PI(p.body.getAngle()) - mod2PI(this.stopPushAngle[rowIdx][i])) < 0.08
                            || Math.abs(distance - this.gripperPushDist[rowIdx][i]) < 5) {
                            p.body.setFixedRotation(true);
                            p.body.setLinearDamping(Infinity);
                            p.setMoveFilter();

                            let collideBot = false;
                            this.world.rayCast(g.bot.getWorldPoint(g.botEdge.m_vertex1),
                                g.bot.getWorldPoint(g.botEdge.m_vertex2),
                                function (fixture, point, normal, fraction) {
                                    if (fixture === p.body.getFixtureList()) {
                                        collideBot = true;
                                        return 0;
                                    }
                                    return -1;
                                });

                            if (collideBot) {
                                g.bot.setLinearVelocity(planck.Vec2(0, 0));
                                p.body.setAngle(this.stopPushAngle[rowIdx][i]);
                            }
                        }
                    } else if (Math.abs(planck.Vec2.lengthOf(planck.Vec2.sub(g.bot.getPosition(), g.botPos))
                        - this.gripperPushDist[rowIdx][i]) < 5) {
                        g.stop();
                    }
                }
            }
        } else if (dt === 350) {
            for (const row of this.polygons) {
                for (const p of row) {
                    p.squeeze();
                    p.body.setFixedRotation(false);
                }
            }

            for (const row of this.polygonPins) {
                for (const p of row) {
                    this.world.destroyJoint(p);
                }
            }

            for (const row of this.grippers) {
                for (const g of row) {
                    g.grasp();
                }
            }
        } else if (350 < dt && dt < 500) {
            // squeeze/grasp phase
            for (const [rowIdx, row] of this.grippers.entries()) {
                for (const [i, g] of row.entries()) {
                    let distance = g.distance();

                    if (i < this.polygons[rowIdx].length) {
                        let p = this.polygons[rowIdx][i];
                        if (Math.abs(mod2PI(p.body.getAngle()) - mod2PI(this.stopSqueezeAngle[rowIdx][i])) < 0.08) {
                            p.body.setFixedRotation(true);
                            p.body.setLinearDamping(Infinity);
                            p.setMoveFilter();

                            let collideTop = false;
                            this.world.rayCast(g.top.getWorldPoint(g.topEdge.m_vertex1),
                                g.top.getWorldPoint(g.topEdge.m_vertex2),
                                function (fixture, point, normal, fraction) {
                                    if (fixture === p.body.getFixtureList()) {
                                        collideTop = true;
                                        return 0;
                                    }
                                    return -1;
                                });

                            if (collideTop) {
                                g.top.setLinearVelocity(planck.Vec2(0, 0));
                                p.body.setAngle(this.stopSqueezeAngle[rowIdx][i]);
                            }
                        }
                    }
                    if (Math.abs(distance - this.gripperSqueezeDist[rowIdx][i]) < 5 || distance < 3) {
                        g.stop();
                        if (i < this.polygons[rowIdx].length) {
                            let p = this.polygons[rowIdx][i]
                            p.body.setAngle(this.stopSqueezeAngle[rowIdx][i]);
                            p.body.setFixedRotation(true);
                            p.body.setLinearDamping(Infinity);
                        }
                    }
                }
            }
        } else if (dt === 500) {
            for (const row of this.polygons) {
                for (const p of row) {
                    p.body.setFixedRotation(true);
                    p.body.setLinearDamping(Infinity);
                }
            }
            for (const row of this.grippers) {
                for (const g of row) {
                    g.unsqueeze();
                }
            }
        } else if (dt > 500) {
            for (const row of this.grippers) {
                for (const g of row) {
                    g.limitUnsqueeze();
                }
            }
        }

        this.stepEngine();
    }
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        updateAnim: function (n, value, state, angles, prev) {
            if (!('squeezeFunc' in state) ||
                !('diameterFunc' in state) ||
                !('radiusFunc' in state) ||
                !('pushFunc' in state) ||
                !('pushGraspFunc' in state) ||
                !('points' in state) ||
                !('sqPlan' in state) ||
                !('pgPlan' in state) ||
                !('convexhull' in state)) {
                return [];
            }

            if (value === "stop") {
                return [[], angles, true, true, prev]
            } else if (value === "sq_anim") {
                let squeezeDisplay = new SqueezeDisplay(state.points, state['sqPlan'], angles.sqPolygonAngles || [],
                    state.diameterFunc, state.squeezeFunc);
                let figs = squeezeDisplay.stepDraw(5);
                angles.sqPolygonAngles = squeezeDisplay.serialize();
                return [figs, angles, false, false, value];
            } else if (value === "pg_anim") {
                let pgDisplay = new PushGraspDisplay(state.points, state.pgPlan, angles.pgPolygonAngles || [],
                    state.radiusFunc, state.diameterFunc, state.pushFunc, state.pushGraspFunc);
                let figs = pgDisplay.stepDraw(5);
                angles.pgPolygonAngles = pgDisplay.serialize();

                return [figs, angles, false, false, value];
            }
        }
    }
});